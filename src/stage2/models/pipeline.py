"""DPM-Solver pipeline for Stage 2 inference."""

import torch
from PIL import Image

from .dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver


class FontDiffuserDPMPipeline:
    """FontDiffuser pipeline with DPM-Solver scheduler for fast sampling."""

    def __init__(
        self,
        model,
        ddpm_train_scheduler,
        version="V3",
        model_type="noise",
        guidance_type="classifier-free",
        guidance_scale=7.5
    ):
        """Initialize the DPM-Solver pipeline.
        
        Args:
            model: FontDiffuserModelDPM instance.
            ddpm_train_scheduler: DDPM scheduler used during training.
            version: Model version string.
            model_type: Type of model prediction ("noise", "x_start", "v").
            guidance_type: Type of guidance ("uncond", "classifier-free").
            guidance_scale: Scale for classifier-free guidance.
        """
        super().__init__()
        self.model = model
        self.train_scheduler_betas = ddpm_train_scheduler.betas
        self.noise_schedule = NoiseScheduleVP(
            schedule='discrete',
            betas=self.train_scheduler_betas
        )

        self.version = version
        self.model_type = model_type
        self.guidance_type = guidance_type
        self.guidance_scale = guidance_scale

    def numpy_to_pil(self, images):
        """Convert numpy images to PIL images.
        
        Args:
            images: Numpy array of images.
            
        Returns:
            List of PIL images.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    def generate(
        self,
        content_images,
        style_images,
        hanzi,
        batch_size,
        order,
        num_inference_step,
        content_encoder_downsample_size,
        t_start=None,
        t_end=None,
        dm_size=(96, 96),
        algorithm_type="dpmsolver++",
        skip_type="time_uniform",
        method="multistep",
        correcting_x0_fn=None,
        generator=None,
    ):
        """Generate images using DPM-Solver.
        
        Args:
            content_images: Content input images.
            style_images: Style reference images.
            hanzi: List of Chinese characters.
            batch_size: Number of images to generate.
            order: Order of DPM-Solver (1, 2, or 3).
            num_inference_step: Number of sampling steps.
            content_encoder_downsample_size: Downsample size for content encoder.
            t_start: Starting time for sampling.
            t_end: Ending time for sampling.
            dm_size: Output image size.
            algorithm_type: DPM-Solver algorithm type.
            skip_type: Time step skip type.
            method: Sampling method ("singlestep" or "multistep").
            correcting_x0_fn: Optional x0 correction function.
            generator: Random number generator.
            
        Returns:
            Tuple of (output_images, intermediate_images).
        """
        model_kwargs = {}
        model_kwargs["version"] = self.version
        model_kwargs["content_encoder_downsample_size"] = content_encoder_downsample_size

        # Prepare conditional inputs
        cond = [content_images, style_images, hanzi]

        # Prepare unconditional inputs for classifier-free guidance
        uncond_content_images = torch.ones_like(content_images).to(self.model.device)
        uncond_style_images = torch.ones_like(style_images).to(self.model.device)
        uncond = [uncond_content_images, uncond_style_images, hanzi]

        # Create model wrapper for DPM-Solver
        model_fn = model_wrapper(
            model=self.model,
            noise_schedule=self.noise_schedule,
            model_type=self.model_type,
            model_kwargs=model_kwargs,
            guidance_type=self.guidance_type,
            condition=cond,
            unconditional_condition=uncond,
            guidance_scale=self.guidance_scale
        )

        # Initialize DPM-Solver
        dpm_solver = DPM_Solver(
            model_fn=model_fn,
            noise_schedule=self.noise_schedule,
            algorithm_type=algorithm_type,
            correcting_x0_fn=correcting_x0_fn
        )

        # Generate initial noise
        x_T = torch.randn(
            (batch_size, 3, dm_size[0], dm_size[1]),
            generator=generator,
        )
        x_T = x_T.to(self.model.device)

        # Run sampling
        x_sample, inter = dpm_solver.sample(
            x=x_T,
            steps=num_inference_step,
            order=order,
            skip_type=skip_type,
            method=method,
        )

        # Post-process output
        x_sample = (x_sample / 2 + 0.5).clamp(0, 1)
        x_sample = x_sample.cpu().permute(0, 2, 3, 1).numpy()
        x_images = self.numpy_to_pil(x_sample)

        # Process intermediate results
        inter_images = []
        for tensor in inter:
            tensor = (tensor / 2 + 0.5).clamp(0, 1)
            tensor = tensor.cpu().permute(0, 2, 3, 1).numpy()
            pil_image = self.numpy_to_pil(tensor)
            inter_images.append(pil_image)

        return x_images, inter_images

