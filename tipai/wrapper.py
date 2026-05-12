from JailbreakDiffusionBench.jailbreak_diffusion.attack import AttackerFactory
from JailbreakDiffusionBench.jailbreak_diffusion.diffusion_model import DiffusionFactory

# Initialize a diffusion model
import sys
import time
sys.path.insert(0, "C:/D-Drive/New Volume/TiPAI-TSPO-inference/tipai")  # adjust this

from pipeline import SafeDiffusionPipeline
from utils.config_loader import load_config
from JailbreakDiffusionBench.jailbreak_diffusion.diffusion_model.core.outputs import GenerationInput, GenerationOutput


class SafeDiffusionWrapper:
    def __init__(self, model_name: str, device: str = "cuda", config_path: str = None):
        if config_path is None:
            raise ValueError("config_path must be provided for SafeDiffusionWrapper")
        self.model_name = model_name
        cfg = load_config(config_path)
        self.pipeline = SafeDiffusionPipeline(cfg)

    def generate(self, input_data: GenerationInput) -> GenerationOutput:
        all_images = []
        start_time = time.time()

        for prompt in input_data.prompts:
            result = self.pipeline.generate(prompt)
            all_images.append(result.image)

        return GenerationOutput(
            images=all_images,
            metadata={
                "model": self.model_name,
                "generation_time": time.time() - start_time
            }
        )