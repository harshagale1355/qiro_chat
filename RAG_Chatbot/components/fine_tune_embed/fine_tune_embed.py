import logging
import os
import random
from dataclasses import dataclass
from typing import Optional, List, Tuple

from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.sentence_transformer import evaluation, losses
from torch.utils.data import DataLoader

# Setup Logging
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FineTuneConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    output_path: str = "./fine_tuned_model"
    epochs: int = 10
    batch_size: int = 8
    warmup_steps: int = 10
    eval_split: float = 0.2
    seed: int = 42

# --- DATASET: Expanded with real snippet data to ensure variation ---
RAW_EXAMPLES: List[Tuple[str, str, float]] = [
    # POSITIVE PAIRS (1.0)
    ("What is the global impact of diabetes?", "Globally, an estimated 422 million adults were living with diabetes in 2014...", 1.0),
    ("What are the health risks of diabetes?", "Diabetes can lead to complications including heart attack, stroke, kidney failure...", 1.0),
    ("How many people die from HAT?", "Latest WHO estimates put HAT cases at 300,000–500,000 with 100,000 dying every year...", 1.0),
    ("What percentage of deaths were CVD in 1900?", "At the beginning of the 20th century, CVD was responsible for less than 10 percent of all deaths...", 1.0),
    ("What is the leading cause of death worldwide?", "Cardiovascular disease (CVD) is the number one cause of death worldwide...", 1.0),
    ("What are neglected tropical diseases?", "NTDs are a group of 13 infections caused by parasitic worms, protozoa or bacteria...", 1.0),
    ("How is hypertension described?", "Hypertension is a serious chronic condition that occurs when the pressure in blood vessels is too high...", 1.0),
    ("Who do NTDs strike?", "They strike the world's poorest people, living in remote and rural areas of low-income countries...", 1.0),
    ("What is Ischaemic heart disease?", "Ischaemic heart disease is the single largest cause of death in developed countries...", 1.0),
    ("How are STHs controlled?", "Control focuses on population-based chemotherapy targeting school-age children...", 1.0),
    ("What is the burden of STHs?", "Over a billion people in the tropics are estimated to be infected with STHs...", 1.0),
    ("What defines rheumatic heart disease?", "RHD is the consequence of acute rheumatic fever...", 1.0),

    # NEGATIVE PAIRS (0.0) - More of these fix the 'nan' issue
    ("What is the impact of diabetes?", "RHD is the consequence of acute rheumatic fever...", 0.0),
    ("How is HAT treated?", "Cardiovascular disease (CVD) is the number one cause of death worldwide...", 0.0),
    ("What are the risks of CVD?", "Soil-transmitted helminths include roundworms, hookworms and whipworms...", 0.0),
    ("Is diabetes expensive?", "Stroke is caused by a disruption in blood flow to part of the brain...", 0.0),
    ("What is a stroke?", "Early diagnosis is the starting point for living well with diabetes...", 0.0),
    ("What is the prevalence of CVD?", "Control focuses on population-based chemotherapy targeting school-age children...", 0.0),
    ("What is the life cycle of HAT?", "High blood pressure increases risk of death even when systolic is 115–130 mmHg...", 0.0),
    ("How to control hypertension?", "NTDs account for approximately one-quarter of the global disease burden from HIV/AIDS...", 0.0),
    ("What causes diabetes?", "CVD was responsible for less than 10% of all deaths worldwide at the start of the 20th century...", 0.0),
    ("What is the DALY of NTDs?", "The percentage of adults with hypertension taking medication increased from 22% to 42%...", 0.0),
]

def train_val_split(examples: List[InputExample], eval_fraction: float, seed: int):
    rng = random.Random(seed)
    # Group by label to ensure a balanced split
    pos = [e for e in examples if e.label == 1.0]
    neg = [e for e in examples if e.label == 0.0]
    
    rng.shuffle(pos)
    rng.shuffle(neg)
    
    # Take 3 from each for validation (Total 6)
    val = pos[:3] + neg[:3]
    train = pos[3:] + neg[3:]
    return train, val

class FineTuner:
    def __init__(self, config: Optional[FineTuneConfig] = None):
        self.config = config or FineTuneConfig()

    def fine_tune(self):
        cfg = self.config
        model = SentenceTransformer(cfg.model_name)
        
        all_examples = [InputExample(texts=[q, a], label=lbl) for q, a, lbl in RAW_EXAMPLES]
        train_examples, val_examples = train_val_split(all_examples, cfg.eval_split, cfg.seed)
        
        logger.info(f"Train size: {len(train_examples)}, Val size: {len(val_examples)}")
        
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=cfg.batch_size)
        train_loss = losses.CosineSimilarityLoss(model)
        
        # Build Evaluator
        evaluator = evaluation.EmbeddingSimilarityEvaluator(
            [ex.texts[0] for ex in val_examples],
            [ex.texts[1] for ex in val_examples],
            [ex.label for ex in val_examples]
        )

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=cfg.epochs,
            output_path=cfg.output_path,
            save_best_model=True
        )
        return SentenceTransformer(cfg.output_path)

if __name__ == "__main__":
    tuner = FineTuner()
    tuner.fine_tune()
    logger.info("Fine-tuning complete. Check the 'fine_tuned_model' folder.")