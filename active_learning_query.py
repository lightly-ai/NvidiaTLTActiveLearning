import os
import argparse

from lightly.api import ApiWorkflowClient
from lightly.active_learning.agents import ActiveLearningAgent
from lightly.active_learning.config import SamplerConfig
from lightly.active_learning.scorers import ScorerObjectDetection

import oracle
import helpers


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, default='',
                        help='API Access Token from the Lightly web-app')
    parser.add_argument('--dataset_id', type=str, default='',
                        help='Id of the dataset in the web-app.')
    parser.add_argument('--preselected_tag_name', type=str, default=None,
                        help='Name of the tag of preselected samples')
    parser.add_argument('--new_tag_name', type=str, default='initial-selection',
                        help='Name of the new tag in the web-app')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Total number of samples after the query')
    parser.add_argument('--method', type=str, default='CORESET',
                        help='Sampling method from [CORESET, RANDOM, CORAL]')
    parser.add_argument('--inferences', type=str, default='./infer_labels',
                        help='Path to the inferred labels')

    args = parser.parse_args()

    # create an api client
    client = ApiWorkflowClient(
        token=args.token,
        dataset_id=args.dataset_id,
    )

    # create an active learning agent
    al_agent = ActiveLearningAgent(
        client,
        preselected_tag_name=args.preselected_tag_name,
    )

    # create a sampler configuration
    config = SamplerConfig(
        n_samples=args.n_samples,
        method=args.method,
        name=args.new_tag_name,
    )

    # create a scorer if inferences exist
    scorer = None
    if os.path.isdir(args.inferences) and args.preselected_tag_name is not None:
        model_outputs = helpers.load_model_outputs(al_agent, args.inferences)
        scorer = ScorerObjectDetection(model_outputs)

    # make an active learning query
    al_agent.query(config, scorer)

    # simulate annotation step by copying the data to the data/train directory
    oracle.annotate_images(al_agent.added_set)
