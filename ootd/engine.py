import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import mlflow
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from sklearn.model_selection import KFold
import logging
import random
import numpy as np

from pipelines_ootd.pipeline_ootd import OotdPipeline
from noise_schedule import AuxiliaryLatent
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from diffusers.models import AutoencoderKL

from pipelines_ootd.unet_garm_2d_condition import UNetGarm2DConditionModel
from pipelines_ootd.unet_vton_2d_condition import UNetVton2DConditionModel
# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OOTDiffusionTrainer(pl.LightningModule):
    def __init__(self, config, train_dataset=None, val_dataset=None):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']

        # Initialize the OotdPipeline
        vae = AutoencoderKL.from_pretrained(VAE_PATH)
        text_encoder = CLIPTextModel.from_pretrained(TEXT_ENCODER_PATH)
        tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_PATH)
        unet_garm = UNetGarm2DConditionModel(**config['model_config'])
        unet_vton = UNetVton2DConditionModel(**config['model_config'])
        scheduler = AuxiliaryLatent(
            in_channels=config['model_config']['in_channels'],
            aux_channels=config['model_config']['aux_channels'],
            out_channels=config['model_config']['out_channels'],
            n_time_step=config['model_config']['n_time_step'],
            latent_type=config['model_config']['latent_type'],
            parameterization=config['model_config']['parameterization'],
            model_config=config['model_config'],
            first_stage_model=vae,
            cond_stage_model=text_encoder,
        )
        safety_checker = None
        feature_extractor = None

        self.pipeline = OotdPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet_garm=unet_garm,
            unet_vton=unet_vton,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

        # Move the pipeline to the specified GPU
        # self.pipeline.to(f"cuda:{config['gpu_id']}")
        self.pipeline.to(self.device)

    def forward(self, prompt, image_garm, image_vton, mask, image_ori):
        return self.pipeline(
            prompt=prompt,
            image_garm=image_garm,
            image_vton=image_vton,
            mask=mask,
            image_ori=image_ori,
            num_inference_steps=100,
            guidance_scale=7.5,
            image_guidance_scale=1.5,
        )

    def training_step(self, batch, batch_idx):
        prompt = batch['prompt']
        image_garm = batch['image_garm']
        image_vton = batch['image_vton']
        mask = batch['mask']
        image_ori = batch['image_ori']
        target_image = batch['target_image']

        # Generate images using the pipeline
        generated_images = self(prompt, image_garm, image_vton, mask, image_ori)

        # Compute the reconstruction loss (consider with diffusion loss and vlb loss)
        reconstruction_loss = self.calculate_reconstruction_loss(generated_images, target_image)

        # Compute the variational lower bound loss from AuxiliaryLatent
        vlb_loss = self.pipeline.scheduler.compute_vlb_loglikehood(generated_images, batch) 

        # Combine the losses
        total_loss = reconstruction_loss + vlb_loss

        # Logging the losses
        self.log('train_reconstruction_loss', reconstruction_loss)
        self.log('train_vlb_loss', vlb_loss)
        self.log('train_total_loss', total_loss)

        # Log generated images to MLflow
        mlflow.log_image(generated_images.detach().cpu().numpy(), "generated_images.png")

        return total_loss

    def validation_step(self, batch, batch_idx):
        prompt = batch['prompt']
        image_garm = batch['image_garm']
        image_vton = batch['image_vton']
        mask = batch['mask']
        image_ori = batch['image_ori']
        target_image = batch['target_image']

        # Generate images using the pipeline
        generated_images = self(prompt, image_garm, image_vton, mask, image_ori)

        # Compute the reconstruction loss
        reconstruction_loss = self.calculate_reconstruction_loss(generated_images, target_image)

        # Compute the variational lower bound loss from AuxiliaryLatent
        vlb_loss = self.pipeline.scheduler.compute_vlb_loglikehood(generated_images, batch)

        # Combine the losses
        total_loss = reconstruction_loss + vlb_loss

        # Logging the validation losses
        self.log('val_reconstruction_loss', reconstruction_loss)
        self.log('val_vlb_loss', vlb_loss)
        self.log('val_total_loss', total_loss)

        # Compute evaluation metrics (e.g., Inception Score, FID)
        inception_score = self.calculate_inception_score(generated_images)
        fid = self.calculate_fid(generated_images, target_image)

        # Log evaluation metrics
        self.log('val_inception_score', inception_score)
        self.log('val_fid', fid)

        # Return the validation loss for Ray Tune
        return {'val_loss': total_loss}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.pipeline.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def calculate_reconstruction_loss(self, generated_images, target_images):
        # Compute the reconstruction loss between generated and target images
        # You can use any appropriate loss function here, such as L1 loss or MSE loss
        reconstruction_loss = nn.L1Loss()(generated_images, target_images)
        return reconstruction_loss

    def calculate_inception_score(self, generated_images):
        # Implement the calculation of Inception Score
        # You can use pre-trained Inception models and compute the score based on the generated images
        # Return the computed Inception Score
        pass

    def calculate_fid(self, generated_images, target_images):
        # Implement the calculation of Fréchet Inception Distance (FID)
        # You can use pre-trained Inception models and compare the distributions of generated and target images
        # Return the computed FID
        pass

# Hyperparameter tuning with Ray Tune
def tune_ootd_diffusion(config):
    # Create the model with the sampled hyperparameters
    model = OOTDiffusionTrainer(config)

    # Create a PyTorch Lightning Trainer with Ray Tune callback
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator = 'gpu',
        devices=config['num_gpus'],
        strategy='ddp',
        callbacks=[
            TuneReportCallback(
                {
                    'val_loss': 'val_loss',
                    'val_inception_score': 'val_inception_score',
                    'val_fid': 'val_fid',
                },
                on='validation_end',
            )
        ]
    )

    # Start training
    trainer.fit(model)

# Usage example
model_config = {
    'in_channels': ...,
    'aux_channels': ...,
    'out_channels': ...,
    'n_time_step': ...,
    'latent_type': ...,
    'parameterization': ...,
    # Add other necessary model configurations
}

# Define the search space for hyperparameters
config = {
    'batch_size': tune.choice([4, 8, 16]),
    'learning_rate': tune.loguniform(1e-5, 1e-3),
    'model_config': model_config,
    'num_gpus': tune.choice([1, 2, 4]),
}

# Perform hyperparameter tuning with Ray Tune
analysis = tune.run(
    tune_ootd_diffusion,
    config=config,
    num_samples=20,
    resources_per_trial={'gpu': 1},
)

# Get the best hyperparameters
best_config = analysis.get_best_config(metric='val_loss', mode='min')
batch_size = best_config['batch_size']
learning_rate = best_config['learning_rate']

# Perform k-fold cross-validation
k = 5
kfold = KFold(n_splits=k)
for fold, (train_indices, val_indices) in enumerate(kfold.split(train_dataset)):
    print(f"Fold {fold+1}")

    # Create data loaders for the current fold
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    # Create the model with the best hyperparameters
    model = OOTDiffusionTrainer(
        config=best_config,
        train_dataset=train_subset,
        val_dataset=val_subset,
    )

    # Create a PyTorch Lightning Trainer with logging and checkpointing
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='checkpoints',
        filename='ootd-diffusion-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        period=1,
    )

    # Set up MLflow experiment tracking
    mlflow.set_experiment("ootd_diffusion_experiment")
    mlflow.start_run(run_name=f"fold_{fold+1}")
    mlflow.log_params(model_config)
    mlflow.log_params(best_config)

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='gpu',
        devices=best_config['num_gpus'],
        strategy='ddp',
        callbacks=[checkpoint_callback],
        logger=pl.loggers.MLFlowLogger(),
    )

    try:
        # Start training
        trainer.fit(model)
    except Exception as e:
        logging.exception("An error occurred during training")
        mlflow.end_run(status='FAILED')
        raise e

    mlflow.end_run(status='FINISHED')









# convert datatype 16 or 32 


    # def convert_to_fp16(self):
    #     """
    #     Convert the torso of the model to float16.
    #     """
    #     self.input_blocks.apply(convert_module_to_f16)
    #     self.middle_block.apply(convert_module_to_f16)

    # def convert_to_fp32(self):
    #     """
    #     Convert the torso of the model to float32.
    #     """
    #     self.input_blocks.apply(convert_module_to_f32)
    #     self.middle_block.apply(convert_module_to_f32)