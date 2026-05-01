# Notice of Code Development


1. ARLArena/verl/workers/actor/dp_actor.py 强制开启calculate_entropy
```
entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=True
                    )
```