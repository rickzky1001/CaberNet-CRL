This repository contains the code for the paper *"CaberNet: Causal Representation Learning for Cross-Domain HVAC Energy Prediction"*, which has been **accepted as a full paper at ACM e-Energy 2026**.

The paper is now available at https://arxiv.org/abs/2511.06634.
## Overview
This repository contains the implementation of CaberNet and several benchmark models for energy prediction and causal inference experiments.

## File Structure

-  **main.py**  
  Change 'train_dic' and 'test_dic' variables for different train and test set split.

-  **Quick Run**

    training and test CaberNet:
    ```bash
    python main.py --model my_model --LLO 0 --train domain_wise --lr 5e-5 --epochs 500
    ```
    training and test lstm (baseline):
    ```bash
    python main.py --model lstm --LLO 0 --train normal --lr 5e-5 --epochs 500
    ```

- **model/**  
  - `my_model.py` and `cond_nn.py`: Implementation of CaberNet.  
  - `lstm.py`: baseline.

- **utils/**  
  - `train.py`: Training pipeline.  

- **result/**  
  Contains results from training and causal inference experiments.  

## Notes
- Due to commercial restrictions, our original HVAC datasets cannot be publicly shared at this time. Before running, please adjust the files related to data processing and data engineering to the correct format.
- Feel free to post a issue or connect if there are any quesions:
rickzky1001@163.com
