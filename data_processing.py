"""
Data processing utilities for WikiText dataset.
Handles tokenization, MLM data preparation, and DataLoader creation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
from typing import Dict, List, Optional, Tuple
import random
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WikiTextMLMDataset(Dataset):
    """Dataset for Masked Language Modeling on WikiText."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512,
        mlm_probability: float = 0.15,
        whole_word_mask: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.whole_word_mask = whole_word_mask
        
        # Tokenize all texts
        logger.info(f"Tokenizing {len(texts)} documents...")
        self.examples = []
        
        for text in tqdm(texts, desc="Tokenizing"):
            if len(text.strip()) > 0:
                # Tokenize and chunk into max_length sequences
                tokens = tokenizer(
                    text,
                    add_special_tokens=False,
                    truncation=False,
                    return_attention_mask=False
                )['input_ids']
                
                # Split into chunks of max_length
                for i in range(0, len(tokens), max_length - 2):  # -2 for CLS and SEP
                    chunk = tokens[i:i + max_length - 2]
                    if len(chunk) > 10:  # Skip very short sequences
                        self.examples.append(chunk)
        
        logger.info(f"Created {len(self.examples)} training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.examples[idx].copy()
        
        # Add special tokens
        tokens = [self.tokenizer.cls_token_id] + tokens + [self.tokenizer.sep_token_id]
        
        # Create attention mask
        attention_mask = [1] * len(tokens)
        
        # Pad to max_length
        padding_length = self.max_length - len(tokens)
        tokens = tokens + [self.tokenizer.pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        
        # Create MLM labels
        tokens, labels = self.mask_tokens(tokens, attention_mask)
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    
    def mask_tokens(self, tokens: List[int], attention_mask: List[int]) -> Tuple[List[int], List[int]]:
        """Apply masking for MLM."""
        labels = [-100] * len(tokens)  # -100 is ignored in loss
        
        # Get positions to mask (excluding special tokens and padding)
        maskable_indices = [
            i for i in range(len(tokens))
            if attention_mask[i] == 1 and tokens[i] not in [
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
                self.tokenizer.pad_token_id
            ]
        ]
        
        # Sample positions to mask
        num_to_mask = max(1, int(len(maskable_indices) * self.mlm_probability))
        mask_indices = random.sample(maskable_indices, min(num_to_mask, len(maskable_indices)))
        
        for idx in mask_indices:
            labels[idx] = tokens[idx]
            
            # 80% of the time, replace with [MASK]
            if random.random() < 0.8:
                tokens[idx] = self.tokenizer.mask_token_id
            # 10% of the time, replace with random token
            elif random.random() < 0.5:
                tokens[idx] = random.randint(0, self.tokenizer.vocab_size - 1)
            # 10% of the time, keep original token
            # (implicitly handled by not changing tokens[idx])
        
        return tokens, labels


def load_wikitext_dataset(
    dataset_name: str = "wikitext-103-v1",
    tokenizer_name: str = "bert-base-uncased",
    max_length: int = 512,
    mlm_probability: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 4,
    cache_dir: Optional[str] = "./cache"
) -> Tuple[DataLoader, DataLoader, DataLoader, AutoTokenizer]:
    """
    Load WikiText dataset and create DataLoaders.
    
    Args:
        dataset_name: WikiText variant to use
        tokenizer_name: HuggingFace tokenizer to use
        max_length: Maximum sequence length
        mlm_probability: Probability of masking tokens
        batch_size: Batch size for DataLoaders
        num_workers: Number of workers for DataLoaders
        cache_dir: Directory to cache dataset
    
    Returns:
        train_loader, val_loader, test_loader, tokenizer
    """
    logger.info(f"Loading {dataset_name} dataset...")
    
    # Load dataset
    if dataset_name == "wikitext-103-v1":
        dataset = load_dataset("wikitext", "wikitext-103-v1", cache_dir=cache_dir)
    elif dataset_name == "wikitext-2-v1":
        dataset = load_dataset("wikitext", "wikitext-2-v1", cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
    
    # Ensure tokenizer has necessary tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Extract texts
    train_texts = [example['text'] for example in dataset['train'] if example['text'].strip()]
    val_texts = [example['text'] for example in dataset['validation'] if example['text'].strip()]
    test_texts = [example['text'] for example in dataset['test'] if example['text'].strip()]
    
    logger.info(f"Train texts: {len(train_texts)}, Val texts: {len(val_texts)}, Test texts: {len(test_texts)}")
    
    # Create datasets
    train_dataset = WikiTextMLMDataset(
        train_texts, tokenizer, max_length, mlm_probability
    )
    val_dataset = WikiTextMLMDataset(
        val_texts, tokenizer, max_length, mlm_probability
    )
    test_dataset = WikiTextMLMDataset(
        test_texts, tokenizer, max_length, mlm_probability
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created DataLoaders - Train batches: {len(train_loader)}, "
                f"Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, tokenizer


def create_pretraining_dataloader(
    texts: List[str],
    tokenizer,
    max_length: int = 512,
    batch_size: int = 32,
    num_workers: int = 4,
    mlm_probability: float = 0.15
) -> DataLoader:
    """Create a DataLoader for pretraining from raw texts."""
    dataset = WikiTextMLMDataset(
        texts, tokenizer, max_length, mlm_probability
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )


if __name__ == "__main__":
    # Test data loading
    logger.info("Testing data loading...")
    
    # Use smaller dataset for testing
    train_loader, val_loader, test_loader, tokenizer = load_wikitext_dataset(
        dataset_name="wikitext-2-v1",
        batch_size=4,
        max_length=128
    )
    
    # Test one batch
    batch = next(iter(train_loader))
    logger.info(f"Batch keys: {batch.keys()}")
    logger.info(f"Input shape: {batch['input_ids'].shape}")
    logger.info(f"Labels shape: {batch['labels'].shape}")
    logger.info(f"Attention mask shape: {batch['attention_mask'].shape}")
    
    # Decode some examples
    for i in range(min(2, batch['input_ids'].shape[0])):
        input_ids = batch['input_ids'][i]
        labels = batch['labels'][i]
        
        # Get masked positions
        masked_positions = (labels != -100).nonzero(as_tuple=True)[0]
        
        logger.info(f"\nExample {i}:")
        logger.info(f"Input: {tokenizer.decode(input_ids[:50])}")  # First 50 tokens
        logger.info(f"Masked positions: {masked_positions[:10].tolist()}")  # First 10 masked