import annoy
from flask import Flask, request, jsonify
import logging
import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__)
logger = logging.getLogger('werkzeug')
logger.setLevel(logging.ERROR)


@app.route('/', methods=['GET'])
def general():
    return 'Hackaton chat model'


@app.route('/question', methods=['POST'])
def get_answer():

    global idx

    request_json = request.get_json()
    text = request_json['question']
    data = {'success': False}

    if text.isdigit() and idx is not None:
        if int(text) > df_1['n_answers'].iloc[idx]:
            answer = 'Данный индекс не входит в диапазон.'

        elif int(text) == 0:
            answer = 'Я вас слушаю.'
        else:
            try:
                requirement = re.findall(f'({int(text)}\) )(.*)\n\n\n', df_1.iloc[idx]['Answer'])[0][1]
                answer = 'НПА про ' + requirement.lower()[:-1] + ':' + '\n\n' + df_2.loc[df_2['Question'] ==
                                                                                         requirement, 'Answer'].iloc[0]
            except IndexError:
                answer = 'Что-то пошло не так. Попробуйте ещё раз.'

        data['success'] = True
        data['answer'] = answer

        return jsonify(data)

    sentence = embed_sentence(text)
    index, distance = annoy_index.get_nns_by_vector(sentence, 1, include_distances=True)
    if distance[0] > 0.68:
        answer = 'К сожалению, у меня нет ответа на этот вопрос. Пожалуйста, постарайтесь переформулировать ' \
                 'его или запишитесь на консультацию.'
    else:
        idx = index[0]
        answer = df_1['Answer'].iloc[idx] + f'Могу рассказать о нормативно-правовых актах по любому из ' \
                                            f'этих требований. Введите число от 1 до ' \
                                            f'{df_1["n_answers"].iloc[idx]} для выбора требования. Либо ' \
                                            f'0 для возврата на предыдущий шаг.'

        data['success'] = True
        
    data['answer'] = answer

    return jsonify(data)


def embed_sentence(sentence):
    text = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt').to(model.device)

    with torch.inference_mode():
        model_output = model(**text)

    token_embeddings = model_output[0]
    expanded_mask = text['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()

    return (torch.sum(token_embeddings * expanded_mask, dim=1) /
            torch.clamp(expanded_mask.sum(dim=1), min=1e-9)).squeeze()


if __name__ == '__main__':
    df_1 = pd.read_csv('./qa_dataset_1.csv')
    df_2 = pd.read_csv('./qa_dataset.csv')
    model_name = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModel.from_pretrained(model_name).to(device)
    annoy_index = annoy.AnnoyIndex(model.config.hidden_size, 'angular')
    annoy_index.load('./models/qa_annoy_model_1.ann')
    idx = None

    app.run(host='0.0.0.0', port=8080)
