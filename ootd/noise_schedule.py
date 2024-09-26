import torch 
import torch.nn as nn
import numpy as np
# import jax.numpy as jnp
from scipy import intergate,solve_ivp
import warnings
from typing import Union
from torch.distributions import RelaxedOneHotCategorical    

from utils import extract_into_tensor, timestep_embedding, DiagonalGaussianDistribution
from torch.autograd import Function

class ScheduleFunction(Function):
    @staticmethod
    def forward(ctx, aux_a, aux_b, aux_c, timesteps):
        ctx.save_for_backward(aux_a, aux_b, aux_c, timesteps)
        
        def function_f(t):
            return aux_a.pow(2)*t.pow(5)/5 + aux_a*aux_c*t.pow(4)/2 + (aux_b.pow(2) +2*aux_a*aux_c)*t.pow(3)/3 + aux_b*aux_c*t.pow(2) + aux_c.pow(2)*t
        
        schedule_function_min = -13.3
        schedule_function_max = 5
        gamma_values = []
        for t in timesteps:
            gamma_value = schedule_function_min + (schedule_function_max - schedule_function_min)*function_f(t)/function_f(torch.tensor(1.0))
            gamma_values.append(gamma_value)
        
        gamma = torch.stack(gamma_values, dim=0)
        alpha = torch.sqrt(torch.sigmoid(-gamma))
        sigma = torch.sigmoid(gamma)
        noise_schedule = torch.sigmoid(-gamma)/torch.sigmoid(gamma)
        
        return alpha, sigma, gamma, noise_schedule

    @staticmethod
    def backward(ctx, grad_alpha, grad_sigma, grad_gamma, grad_noise_schedule):
        aux_a, aux_b, aux_c, timesteps = ctx.saved_tensors
        
        def function_f(t):
            return aux_a.pow(2)*t.pow(5)/5 + aux_a*aux_c*t.pow(4)/2 + (aux_b.pow(2) +2*aux_a*aux_c)*t.pow(3)/3 + aux_b*aux_c*t.pow(2) + aux_c.pow(2)*t
        
        # Compute gradients for aux_a, aux_b, aux_c
        grad_aux_a = torch.zeros_like(aux_a)
        grad_aux_b = torch.zeros_like(aux_b)
        grad_aux_c = torch.zeros_like(aux_c)
        
        for t in timesteps:
            df_da = 2*aux_a*t.pow(5)/5 + aux_c*t.pow(4)/2 + 2*aux_c*t.pow(3)/3
            df_db = 2*aux_b*t.pow(3)/3 + aux_c*t.pow(2)
            df_dc = aux_a*t.pow(4)/2 + 2*aux_a*t.pow(3)/3 + aux_b*t.pow(2) + 2*aux_c*t
            
            grad_aux_a += grad_gamma * df_da
            grad_aux_b += grad_gamma * df_db
            grad_aux_c += grad_gamma * df_dc
        
        return grad_aux_a, grad_aux_b, grad_aux_c, None
class AuxiliaryNoise(nn.Module):
    def init(
            self,
            in_channels,
            aux_channels,
            out_channels
    ):
        super().init()
        self.in_channels = in_channels
        self.aux_channels = aux_channels
        self.out_channels = out_channels
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, aux_channels),
        )

    def forward(self,x):
        return self.mlp(x)

class AuxiliaryLatent(nn.Module):
    def init(self,
                 in_channels,
                 aux_channels, #3072
                 out_channels,
                 n_time_step,
                 latent_type,
                 parameterization,
                 model_config,
                 first_stage_model,
                 cond_stage_model,
                 first_stage_key):
        self.in_channels = in_channels
        self.aux_channels = aux_channels # 3072
        self.out_channels = out_channels
        self.first_stage_model = first_stage_model # autoencoderkl
        self.cond_stage_model = cond_stage_model # text encoder
        self.num_timesteps = n_time_step
        self.model = DiffusionWrapper(model_config) #unet config # change to fit OOT
        self.parameterization = parameterization
#latet vector
        self.first_stage_key = first_stage_key
        self.z,_ = self.get_input(batch = self.batch_size, self.first_stage_key,
                                  return_first_stage_outputs=True,
                                  force_c_code = True,
                                  return_orginal_cond = True,
                                  )
        self.latent_type = latent_type
        if self.latent_type == "cls_label":
            self.z = RelaxedOneHotCategorical(self.z)
        elif self.latent_type == "low-dimensional":
            self.z = self.first_stage_model.get_codebook_indices(self.z)
        elif self.latent_type == "text":
            self.z = self.cond_stage_model(self.z)
        else:
            return ValueError("type should be 'cls_label', 'low-dimensional', 'text'")
        
        # adpative schedule
        aux_output = AuxiliaryNoise(self.in_channels, self.aux_channels, self.out_channels)
        self.aux_a, self.aux_b, self.aux_c = aux_output.split(self.aux_channels, dim=-1)
        self.timesteps = torch.linspace(0,1, self.n_time_step, dtype = torch.float64)
        self.alpha, self.sigma, self.gamma,  self.noise_schedule = self.register_schedule_forward()
    def register_schedule_forward(self):
        self.alpha, self.sigma, self.gamma, self.noise_schedule = self.schedule_function(
            self.aux_a, self.aux_b, self.aux_c, self.timesteps
        )
        return self.alpha, self.sigma, self.gamma, self.noise_schedule

    def register_schedule_backward(self, s, t):
        # This method can now use the gradients computed in ScheduleFunction.backward
        alpha_s, sigma_s, _, _ = self.schedule_function(self.aux_a, self.aux_b, self.aux_c, s.unsqueeze(0))
        alpha_t, sigma_t, _, _ = self.schedule_function(self.aux_a, self.aux_b, self.aux_c, t.unsqueeze(0))
        
        backward_mean_coef1 = (alpha_t * sigma_s) / (alpha_s * sigma_t + 1e-8)
        backward_mean_coef2 = (sigma_t - alpha_t.pow(2)/(alpha_s.pow(2)*sigma_s + 1e-8))*alpha_s/sigma_t
        backward_variance = (sigma_s/sigma_t + 1e-8)*(sigma_t - alpha_t.pow(2)/(alpha_s.pow(2)*sigma_s + 1e-8))
        
        return backward_mean_coef1, backward_mean_coef2, backward_variance, alpha_t, sigma_t
    
    def register_schedule_sde(self,t):
        def drift_func(t):
            drift = self.aux_a.pow(2)*t.pow(4) + self.aux_a*self.aux_c*t.pow(3)*2 + (self.aux_b.pow(2) +2*self.aux_a*self.aux_c)*t.pow(2) + self.aux_b*self.aux_c*t*2 + self.aux_c.pow(2)
            return drift
        # rechange it into numberic index
        f_t = -0.5*self.noise_schedule[t]*drift_func(t)
        g2_t = self.noise_schedule[t]*drift_func(t)
        return f_t, g2_t
    def predict_start_from_noise(self, x_t, t, noise):
        # Get the backward schedule parameters
        backward_mean_coef1, backward_mean_coef2, backward_variance, alpha_t, sigma_t = self.register_schedule_backward(t, torch.zeros_like(t))

        # Extract the relevant tensors
        alpha_t = extract_into_tensor(alpha_t, x_t.shape)
        sigma_t = extract_into_tensor(sigma_t, x_t.shape)

        # Predict x_0
        if self.parameterization == "eps":
            x_0 = (x_t - sigma_t * noise) / alpha_t
        elif self.parameterization == "x0":
            x_0 = noise  # In this case, the model directly predicts x_0
        elif self.parameterization == "v":
            x_0 = alpha_t * x_t - sigma_t * noise
        else:
            raise ValueError(f"Unsupported parameterization: {self.parameterization}")

        # Clip x_0 to ensure it's within a valid range (usually [-1, 1] for normalized images)
        x_0 = torch.clamp(x_0, -1.0, 1.0)

        return x_0
    def q_posterior(self,x_recon, x,t,noise):
        
        backward_mean_coef1,backward_mean_coef2,backward_variance = self.register_schedule_backward(self,0,t)
        mean = extract_into_tensor(backward_mean_coef1) * x_recon + extract_into_tensor(backward_mean_coef2)*x
        variance = extract_into_tensor(backward_variance)
        log_variance = extract_into_tensor(torch.log(variance))
        return mean, variance, log_variance
    
    def p_mean_variance(self, x, x_start,t, noise, clip_denoised: bool):
        model_out = self.model(x, t)
        backward_mean_coef1,backward_mean_coef2,backward_variance = self.register_schedule_backward(self,0,t)

        if self.parameterization == 'eps':
            x_recon = extract_into_tensor(backward_mean_coef1) * model_out + extract_into_tensor(backward_mean_coef2)*noise
        if self.parameterization == "x0":
            x_recon = model_out
        if self.parameterization == "velocity":
            x_recon = model_out

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_recon, x, t)
        return model_mean, posterior_variance, posterior_log_variance      
        
    #from here we can compute velocity loss with F_hat (loss_diff 1 and loss_diff 2)
    def velocity_parameterization(self, x_start, noise):
        alpha, sigma,_,noise_schedule = self.register_schedule_forward(self.timesteps)
        x_t = x_start * alpha + sigma * noise
        velocity = (alpha * x_start - self.model(x_t, t)) / (sigma + 1e-8)
        return -x_t - velocity * torch.exp(-0.5 * torch.log(noise_schedule))

    # compute equation 81
    def compute_loglikehood(self, x_start, z):
        def ode_fn(t,x):
            alpha, sigma, = self.register_schedule_forward(t)
            if self.parameterization == 'eps':
                s_xt_z = -self.model(x,t)/sigma
            if self.parameterization == "x0":
                s_xt_z = self.model(x,t)
            if self.parameterization == "velocity":
                s_xt_z = self.velocity_parameterization(x_start,t)
            else:
                raise ValueError("Invalid parameterization")
            
            f_t, g2_t = self.register_schedule_sde(t)

            drift = f_t - 1/2*g2_t*s_xt_z

            return drift
        t_span = [1.0, 0.0]
        sol = solve_ivp(ode_fn, t_span, [x_start], t_eval=torch.linspace(1.0, 0.0, self.timesteps), method='RK45', rtol=1e-5, atol=1e-5)
        log_p0_x1 = self.log_prob(sol.y[:, -1])
        trace_estimator = 0.0

        for i in range(self.num_timesteps - 1):
            t_i = sol.t[i]
            x_i = sol.y[i]
            eps = torch.randn_like(x_i)
            trace_estimator += eps.dot(ode_fn(t_i, x_i))/self.num_timesteps
        return log_p0_x1 - trace_estimator    
    # compute vlb with equation 83
    def compute_vlb_loglikehood(self,x_start,batch,other = None):
        z = self.get_input(x_start)[0]
        encoder_posterior = self.encode_first_stage(x_start)
        if self.latent_type == "continuous":
            if isinstance(encoder_posterior, DiagonalGaussianDistribution):
                kl = encoder_posterior.kl(other = other)
                kl = kl.mean()
            elif isinstance(encoder_posterior, torch.Tensor):
                posterior_mean = torch.mean(encoder_posterior)# refernce to autoencoder from ootpipeline
                posterior_var = torch.var(encoder_posterior)
                log_var = torch.log(posterior_var)
                kl =  0.5*torch.sum(torch.pow(posterior_mean,2) + posterior_var - 1 - log_var, dim = [1,2,3])   
        if self.latent_type == "discrete":
            kl = self.kl_loss_discrete_sog(z,k =20, tau = 1e-5, s=20)

        log_p_z = self.compute_loglikehood(x_start,z)

        return log_p_z - kl
    #compute eq 85
    def compute_importance_weighted_log_likehood(self, x_start, K = 20):
        z = self.get_input(x_start)[0]
        log_likelihoods = []
        for _ in range(K):
            eps = torch.randn_like(x_start)
            dequantized_x_start = x_start + eps
            log_p_z = self.compute_loglikehood(dequantized_x_start,z)
            log_q_eps = torch.sum(-0.5 * eps.pow(2) - 0.5 * torch.log(2 * torch.pi), dim=tuple(range(1, eps.ndim)))
            log_likelihoods.append(log_p_z - log_q_eps)

        log_likelihoods = torch.stack(log_likelihoods, dim = -1)
        return torch.logsumexp(log_likelihoods, dim=-1) - torch.log(torch.tensor(K, device=x_start.device))


    def kl_loss_discrete_sog(w, k, tau, s):
        # Compute the KL loss term for discrete z using SOG distribution
        # w: shape (batch_size, n), logits (unnormalized weights) for each item
        # k: scalar, number of items to sample (subset size)
        # tau: scalar, temperature parameter for SOG distribution
        # s: scalar, number of Gamma distributions to sum in SOG distribution

        # Compute the normalization constant Z
        Z = torch.sum(w, dim=1, keepdim=True)

        # Compute the log probabilities
        log_w = torch.log(w + 1e-8)  # Add a small constant for numerical stability
        log_Z = torch.log(Z + 1e-8)
        log_n = torch.log(torch.tensor(w.shape[1], dtype=torch.float32))

        # Sample from the SOG distribution
        epsilon_sog = torch.sum(torch.distributions.Gamma(1.0 / k, 1.0).sample((s, w.shape[0], w.shape[1])), dim=0) - torch.log(torch.tensor(s, dtype=torch.float32))
        epsilon_sog = tau * epsilon_sog

        # Compute the perturbed logits and apply Top-K sampling
        perturbed_logits = log_w + epsilon_sog
        top_k_indices = torch.topk(perturbed_logits, k, dim=1)[1]

        # Compute the sampled subset using one-hot encoding
        sampled_subset = torch.nn.functional.one_hot(top_k_indices, num_classes=w.shape[1]).float()

        # Compute the KL loss term
        kl_loss = -torch.sum(sampled_subset * (log_w - log_Z - log_n), dim=1)

        # Compute the mean KL loss over the batch
        kl_loss_mean = torch.mean(kl_loss)

        return kl_loss_mean
    
    def set_index(self):
        return self._step_index
    
    @property
    def begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index

    def scale_model_index(
            self, sample: torch.FloatTensor, timestep: torch.FloatTensor,
    )-> torch.FloatTensor:
        if self.step_index is None:
            self._init_step_index()
        alpha,sigma,_ = self.register_schedule_forward(timestep)
        sample = alpha*sample/((sigma**2+1e-8)**0.5)
        self.is_scale_input_called = True

        return sample
    
    def get_lms_coefficient(self, order, t, current_order):
        """
        Compute the linear multistep coefficient.

        Args:
            order (int): The order of the linear multistep method.
            t (int): The current timestep index.
            current_order (int): The current order index.
        """
        s = self.timesteps[t]
        t_next = self.timesteps[t + 1]

        _, _, _, alpha_s, sigma_s = self.register_schedule_backward(s, s)
        _, _, _, alpha_t, sigma_t = self.register_schedule_backward(s, t_next)

        prod = 1.0
        for k in range(order):
            if current_order == k:
                continue
            s_k = self.timesteps[t - k]
            _, _, _, alpha_sk, sigma_sk = self.register_schedule_backward(s_k, s_k)
            prod *= (sigma_t - sigma_sk) / (sigma_s - sigma_sk)

        return prod

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        if device is None:
            device = self.device
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_timesteps // self.num_inference_steps
        timesteps = torch.arange(0, self.num_timesteps, step_ratio, dtype=torch.float32, device=device)
        self.timesteps = timesteps.flip(0)

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonezero()

        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()
    

    def _init_step_index(self, t):
        if self.step_index is None:
            if isinstance(t, torch.Tensor):
                timestep = t.to(self.timesteps.device)
            self.step_index = self.index_for_timestep(t)
            

    def _sigma_to_t(self, sigma):
        log_sigma = torch.log(sigma)
        dists = torch.abs(log_sigma - torch.log(self.sigmas))
        low_idx = torch.argmin(dists)
        high_idx = low_idx + 1
        low = self.sigmas[low_idx]
        high = self.sigmas[high_idx]
        w = (sigma - low) / (high - low)
        t = (1 - w) * self.timesteps[low_idx] + w * self.timesteps[high_idx]
        return t
    def _convert_to_karras(self, t):
        sigma = self.sigma[t]
        log_sigma = torch.log(sigma)
        karras_sigma = torch.exp(log_sigma * self.betas_for_alpha_bar[t])
        return karras_sigma
    def step(self,
             model_output: torch.FloatTensor, #  consider this is x_t
             timestep: Union[float, torch.FloatTensor], #[0,1]
             sample: torch.FloatTensor,
             order: int = 0,
             return_dict: bool = True
             ):
        # ourpose is to return prev_sample and pred_original_sample, basically is final denoising image and it's closest previous one
        if not self.is_scale_input_called:
            warnings.warn(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example." )
        
        if self.step_index is None:
            self._init_step_index(timestep)

        s = self.timesteps[self.step_index]
        t = self.timesteps[self.step_index+1]

        backward_mean_coef1, backward_mean_coef2, backward_variance, alpha_t, sigma_t = self.register_schedule_backward(s,t)
        if self.parameterization == "eps":
            x_0 = sample - sigma_t * model_output
        elif self.parameterization == "x0":
            x_0 = model_output
        elif self.parameterization == "v":
            x_0 = alpha_t * sample - sigma_t * model_output
        else:
            raise ValueError(f"Unsupported parameterization type: {self.parameterization}")
        
        # compute x_0
        eps = model_output #self.model(x_0,t)
        pred_orrignal_output = alpha_t * self.model(sample, t) + sigma_t * eps

        # compute previous x_t-1
        prev_sample = backward_mean_coef1*sample + backward_mean_coef2*pred_orrignal_output

        # ODE derivative

        # derivative = (sample - prev_sample)/(s-t)

        # self.derivative.append(derivative)

        #compute equivalent term off dervivative
        f_t, g2_t = self.register_schedule_sde(t)

        dervivative = (sample - prev_sample)/torch.sqrt(g2_t)
        self.derivative.append(dervivative)

        if len(self.derivative) > order:
            self.derivative.pop(0)

        order = min(self.step_index +1,order)
        lms_coeffs = [self.get_lms_coefficient(order, self.step_index, curr_order) for curr_order in range(order)]

        prev_sample = sample + sum(
            coeff +derivative for coeff, derivative in zip(lms_coeffs, self.derivative)
        )
        # increase  step size for next step
        self._step_index +=1

        if not return_dict:
            return (prev_sample,)
        
        return prev_sample, pred_orrignal_output
        
    def add_noise(self, x_0, noise):        
        return self.alpha*x_0 + self.sigma*noise
