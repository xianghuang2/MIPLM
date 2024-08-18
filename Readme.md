This repository is dedicated to a table data filling algorithm ‘Missing Value Imputation via Pre-trained Language Models with Trainable Prompt and Retrieval Augmentation’, offering several dataset examples and corresponding environment setup.

Our setup environment is notably simple, detailed as follows:

**## Setup Environment**

*```*

conda create --name MIPLM --yes

conda activate MIPLM

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

conda install pandas scikit-learn logging argparse tqdm transformers

*```*

Algorithm parameters can be adjusted in the 'param.json' file, where you can also include parameters for your unique datasets. The key configurations are explained below:

**## Configurations**

"dataset_name": Dataset name "data_path": Dataset location "miss_column": missing columns "cat_column": Collection of categorical columns "num_column": Collection of numerical columns "detail_column": Collection of detail(seq) columns "epochs": train epochs

To run the code, execute the following command:

**## Run an Experiment**

*```*

python main.py 

*```*

In the `Arguments` section, you can set the missing rate in the `program_set.py` file, where the default method is MCAR applied to the 'miss_column.' If you wish to use prompt tuning, you can set `setting['finetune_all']` to `True` in `main.py`. Please note that 'use_annoy' is a module we have not yet developed, so it should be set to `False` by default. If you plan to use the RAG method, you will need to independently retrieve an additional knowledge base beforehand. For fine-grained retrieval, you can refer to the script in `rag_info.py`. For coarse-grained retrieval, you must manually search the additional knowledge base for extra data with matching keys.

For adding new datasets, follow these steps:

**## Adding a Tabular Dataset**

*```*

Add datasets under the 'datasets/data_imputation' folder and configure the parameters in 'param.json'.

*```*

Moreover, you must download LLM and add path at main.py 54 lines.
