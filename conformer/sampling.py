import transformers
import torch
import copy

from conformer.base import CElement, ConformerBase


class Sampler(ConformerBase):
    """
    A class derived from ConformerBase, designed for sampling predictions from a model.
    """

    def __init__(self, model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer, lambda_vector):
        """
        Initialize with a model, tokenizer, and lambda vector.
        """
        super().__init__(model, tokenizer)
        self.lambda_vector = lambda_vector

    @classmethod
    def from_calibrator(cls, calibrator, lambda_vector):
        """
        Creates a full copy of a Calibrator instance as a Sampler instance,
        adding lambda_vector and the sample_with_rejection method.
        """
        # Create a deep copy of the Calibrator instance
        sampler = copy.deepcopy(calibrator)
        
        # Set additional attributes specific to Sampler
        sampler.lambda_vector = lambda_vector
        
        # Add the sample_with_rejection method (defined below)
        sampler.sample_with_rejection = cls.sample_with_rejection.__get__(sampler)
        
        return sampler

    def sample_with_rejection(self, prompt: str, k_max: int) -> torch.Tensor:
        assert self.rejection_functions, "Quality estimator function not set."
        assert self.group_confidence_function, "Group confidence function not set."

        C_lambda = []
        group_conf_idx = self.func_lambda_map[self.group_confidence_function.__name__]

        for _ in range(k_max):
            # tokens = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            tokens = self.tok_func(prompt).to(self.model.device)
            # outputs = self.model.generate(**tokens, return_dict=True)
            outputs = self.model.generate(
                **tokens, 
                max_new_tokens=self.max_calibration_output_length, 
                num_return_sequences=1,
                return_dict_in_generate=True, 
                output_scores=True,
                do_sample=True, 
                top_k=50, 
                top_p=0.95, 
            )

            # Calculate transition scores
            prompt_n = len(tokens["input_ids"][0])
            scores = tuple([outputs.scores[j][0].unsqueeze(0) for j in range(len(outputs.scores))])
            transitions = self.model.compute_transition_scores(outputs.sequences[0][prompt_n:].unsqueeze(0),scores,normalize_logits=True)
            transitions = torch.nan_to_num(transitions, neginf=0.0)

            y_k = CElement(
                prompt=prompt,
                prompt_tokens=tokens.input_ids[0].detach().cpu(),
                response=self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True),
                sequence_score=transitions.sum().detach().cpu() / transitions.size(1),
                transition_scores=transitions.detach().cpu(),
                response_tokens=outputs.sequences[0].detach().cpu()
            )

            # Rejection checks
            for reject_func in self.rejection_functions:
                lambda_idx = self.func_lambda_map[reject_func.__name__]
                if reject_func(prompt, y_k, C_lambda, self.lambda_vector[lambda_idx]):
                    break
            else:
                C_lambda.append(y_k)

                # Check group confidence
                if self.group_confidence_function(prompt, C_lambda, self.lambda_vector[group_conf_idx]):
                    break

        return torch.stack([c.response_tokens for c in C_lambda])  # Convert list of CElements to tensor

# Usage
# Assuming lambdaz is already defined and calibrator is an instance of Calibrator
# sampler = Sampler.from_calibrator(calibrator, lambdaz)
# prompt = 'The Londoner has filmed a TV movie called "My Boy Jack," about author Rudyard Kipling and his son, due for release later this year. Write a spotlight for this:'
# responses = sampler.sample_with_rejection(prompt, 1)
# decoded_responses = [tokenizer.decode(response, skip_special_tokens=True) for response in responses]


