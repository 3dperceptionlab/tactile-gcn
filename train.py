import logging
import sys

import loader.biotacsp_loader

log = logging.getLogger(__name__)

def train():

    CSV_FILE = "biotac-palmdown-grasps.csv"

    biotacsp_dataset_ = loader.biotacsp_loader.BioTacSpDataset(csvFile=CSV_FILE)

    log.info(biotacsp_dataset_)

    for i in range(len(biotacsp_dataset_)):

        sample_ = biotacsp_dataset_[i]
        log.info(sample_)

if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    train()