# script for downloading iwslt2017-de-en (for translating Dutch to English)
from datasets import concatenate_datasets, load_dataset

if __name__ == "__main__":
    # download and combine splits - will do this ourselves later
    iwslt_dataset = load_dataset("IWSLT/iwslt2017", 'iwslt2017-de-en', trust_remote_code=True)
    combined_dataset = concatenate_datasets(list(iwslt_dataset.values()))

    # save to parquet
    combined_dataset.to_parquet("data/raw/iwstl2017-de-en_full.parquet")