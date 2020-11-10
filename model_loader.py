import os
from config import config
from transition_parser_sdp import TransitionParser
from transition_sdp_predictor import SDPParserPredictor
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = '4'

if __name__ == "__main__":
    test_predicted_conlls = []
    model = TransitionParser.load("./models/model_2020_11_07_20_09_02.pt")
    model.metric.reset()

    vocab = model.vocab
    reader = model.reader
    predictor = SDPParserPredictor(model, reader)
    batch_size = 30

    with open(config['test_file_path'], 'r', encoding='utf-8') as fp:
        conlls = fp.read().split('\n\n')[:-1]

    batches = [conlls[i:i + batch_size] for i in range(0, len(conlls), batch_size)]

    print("predict...")
    for sentences in tqdm(batches):
        for result in predictor.predict(sentences):
            for edge in result:
                test_predicted_conlls.append(edge)
            test_predicted_conlls.append('\n')

    print("output...")
    with open(config['test_output_path'].format(model.start_time), 'w', encoding='utf-8') as out:
        out.write(model.metric.__repr__())
        for line in tqdm(test_predicted_conlls):
            out.write(line)
