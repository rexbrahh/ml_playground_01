#!/usr/bin/env python3
"""
Evaluation script to measure model improvement before and after fine-tuning.
"""

import argparse
import torch
import json
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate model performance before and after fine-tuning")
    parser.add_argument(
        "--base_model", 
        type=str, 
        required=True,
        help="Base model name or path"
    )
    parser.add_argument(
        "--fine_tuned_model", 
        type=str, 
        default=None,
        help="Fine-tuned model path (if None, only evaluates base model)"
    )
    parser.add_argument(
        "--test_prompts_file", 
        type=str, 
        default=None,
        help="JSON file with test prompts (if None, uses built-in prompts)"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="evaluation_results.json",
        help="Output file for evaluation results"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=100,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Temperature for generation"
    )
    return parser.parse_args()

def load_test_prompts(file_path=None):
    """Load test prompts from file or use built-in prompts."""
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    
    # Built-in test prompts covering various capabilities
    return [
        {
            "prompt": "Explain quantum computing in simple terms:",
            "category": "explanation",
            "expected_keywords": ["quantum", "bits", "superposition", "computing"]
        },
        {
            "prompt": "Write a Python function to calculate factorial:",
            "category": "code",
            "expected_keywords": ["def", "factorial", "return", "function"]
        },
        {
            "prompt": "List the benefits of renewable energy:",
            "category": "information",
            "expected_keywords": ["renewable", "energy", "environment", "sustainable"]
        },
        {
            "prompt": "Describe the process of photosynthesis:",
            "category": "science",
            "expected_keywords": ["photosynthesis", "plants", "sunlight", "oxygen"]
        },
        {
            "prompt": "What are the key principles of good software design?",
            "category": "technical",
            "expected_keywords": ["design", "software", "principles", "maintainable"]
        },
        {
            "prompt": "How to prepare for a job interview:",
            "category": "advice",
            "expected_keywords": ["interview", "preparation", "questions", "research"]
        },
        {
            "prompt": "Summarize the importance of data privacy:",
            "category": "ethics",
            "expected_keywords": ["privacy", "data", "security", "protection"]
        },
        {
            "prompt": "Explain the difference between machine learning and deep learning:",
            "category": "ai",
            "expected_keywords": ["machine learning", "deep learning", "neural", "algorithms"]
        }
    ]

def load_model_and_tokenizer(model_path, is_peft=False, base_model_path=None):
    """Load model and tokenizer."""
    logger.info(f"Loading model: {model_path}")
    
    # Configure quantization for memory efficiency
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    if is_peft and base_model_path:
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        
        # Load tokenizer from base model
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        
        # Load PEFT adapter
        model = PeftModel.from_pretrained(model, model_path)
        logger.info(f"Loaded PEFT model with base: {base_model_path}")
        
    else:
        # Load complete model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        logger.info(f"Loaded complete model: {model_path}")
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7):
    """Generate response for a given prompt."""
    # Format prompt similar to training format
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the response part
    if "### Response:" in full_response:
        response = full_response.split("### Response:")[-1].strip()
    else:
        response = full_response[len(formatted_prompt):].strip()
    
    return response

def evaluate_response_quality(response, expected_keywords):
    """Simple evaluation of response quality based on keyword presence and length."""
    if not response or len(response.strip()) < 10:
        return {
            "keyword_score": 0.0,
            "length_score": 0.0,
            "overall_score": 0.0,
            "analysis": "Response too short or empty"
        }
    
    # Keyword score: fraction of expected keywords found
    response_lower = response.lower()
    keywords_found = sum(1 for keyword in expected_keywords if keyword.lower() in response_lower)
    keyword_score = keywords_found / len(expected_keywords) if expected_keywords else 0.5
    
    # Length score: reasonable length (50-500 words is good)
    word_count = len(response.split())
    if word_count < 20:
        length_score = 0.3
    elif word_count > 200:
        length_score = 0.7
    else:
        length_score = 1.0
    
    # Overall score
    overall_score = (keyword_score * 0.7 + length_score * 0.3)
    
    analysis = f"Found {keywords_found}/{len(expected_keywords)} keywords, {word_count} words"
    
    return {
        "keyword_score": keyword_score,
        "length_score": length_score,
        "overall_score": overall_score,
        "word_count": word_count,
        "keywords_found": keywords_found,
        "total_keywords": len(expected_keywords),
        "analysis": analysis
    }

def evaluate_model(model, tokenizer, test_prompts, model_name, max_new_tokens=100, temperature=0.7):
    """Evaluate model on test prompts."""
    logger.info(f"Evaluating model: {model_name}")
    results = []
    total_score = 0.0
    
    for i, test_case in enumerate(test_prompts):
        logger.info(f"Processing prompt {i+1}/{len(test_prompts)}: {test_case['category']}")
        
        prompt = test_case["prompt"]
        expected_keywords = test_case.get("expected_keywords", [])
        
        # Generate response
        try:
            response = generate_response(
                model, tokenizer, prompt, 
                max_new_tokens=max_new_tokens, 
                temperature=temperature
            )
            
            # Evaluate response
            evaluation = evaluate_response_quality(response, expected_keywords)
            
            result = {
                "prompt": prompt,
                "response": response,
                "category": test_case["category"],
                "expected_keywords": expected_keywords,
                "evaluation": evaluation
            }
            
            results.append(result)
            total_score += evaluation["overall_score"]
            
            logger.info(f"Score: {evaluation['overall_score']:.2f} - {evaluation['analysis']}")
            
        except Exception as e:
            logger.error(f"Error generating response for prompt {i+1}: {str(e)}")
            results.append({
                "prompt": prompt,
                "response": "ERROR: " + str(e),
                "category": test_case["category"],
                "expected_keywords": expected_keywords,
                "evaluation": {"overall_score": 0.0, "analysis": f"Generation failed: {str(e)}"}
            })
    
    average_score = total_score / len(test_prompts) if test_prompts else 0.0
    
    return {
        "model_name": model_name,
        "average_score": average_score,
        "total_prompts": len(test_prompts),
        "results": results,
        "timestamp": datetime.now().isoformat()
    }

def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load test prompts
    test_prompts = load_test_prompts(args.test_prompts_file)
    logger.info(f"Loaded {len(test_prompts)} test prompts")
    
    evaluation_results = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "test_prompts_count": len(test_prompts),
        "generation_config": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature
        }
    }
    
    # Evaluate base model
    logger.info("="*60)
    logger.info("EVALUATING BASE MODEL")
    logger.info("="*60)
    
    base_model, base_tokenizer = load_model_and_tokenizer(args.base_model)
    base_results = evaluate_model(
        base_model, base_tokenizer, test_prompts, 
        f"base_{args.base_model}",
        args.max_new_tokens, args.temperature
    )
    evaluation_results["base_model"] = base_results
    
    logger.info(f"Base model average score: {base_results['average_score']:.3f}")
    
    # Evaluate fine-tuned model if provided
    if args.fine_tuned_model:
        logger.info("="*60)
        logger.info("EVALUATING FINE-TUNED MODEL")
        logger.info("="*60)
        
        # Check if it's a PEFT model
        is_peft = False
        try:
            peft_config = PeftConfig.from_pretrained(args.fine_tuned_model)
            is_peft = True
            logger.info("Detected PEFT model")
        except:
            logger.info("Detected full model")
        
        fine_tuned_model, fine_tuned_tokenizer = load_model_and_tokenizer(
            args.fine_tuned_model, 
            is_peft=is_peft,
            base_model_path=args.base_model if is_peft else None
        )
        
        fine_tuned_results = evaluate_model(
            fine_tuned_model, fine_tuned_tokenizer, test_prompts,
            f"fine_tuned_{args.fine_tuned_model}",
            args.max_new_tokens, args.temperature
        )
        evaluation_results["fine_tuned_model"] = fine_tuned_results
        
        logger.info(f"Fine-tuned model average score: {fine_tuned_results['average_score']:.3f}")
        
        # Calculate improvement
        improvement = fine_tuned_results['average_score'] - base_results['average_score']
        improvement_percent = (improvement / base_results['average_score']) * 100 if base_results['average_score'] > 0 else 0
        
        evaluation_results["improvement"] = {
            "absolute": improvement,
            "percentage": improvement_percent,
            "analysis": "Improved" if improvement > 0 else "Declined" if improvement < 0 else "No change"
        }
        
        logger.info("="*60)
        logger.info("COMPARISON RESULTS")
        logger.info("="*60)
        logger.info(f"Base model score:       {base_results['average_score']:.3f}")
        logger.info(f"Fine-tuned model score: {fine_tuned_results['average_score']:.3f}")
        logger.info(f"Improvement:            {improvement:+.3f} ({improvement_percent:+.1f}%)")
        logger.info(f"Result:                 {evaluation_results['improvement']['analysis']}")
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    logger.info(f"Evaluation results saved to: {args.output_file}")

if __name__ == "__main__":
    main()