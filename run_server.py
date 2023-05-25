import annoy
from flask import Flask, request, jsonify
import logging
import pandas as pd
import pickle
import re
import torch
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__)
logger = logging.getLogger('werkzeug')
logger.setLevel(logging.ERROR)


@app.route('/', methods=['GET'])
def general():
    return 'Hackaton chat model'


@app.route('/question', methods=['POST'])  # Принимает JSON вида {'id': str, 'question': str, 'new_chat': bool}
def get_answer():                          # Возвращает JSON вида {'answer': str, 'success': bool}
    global client_info

    request_json = request.get_json()
    text = request_json['question']
    user_id = request_json['id']
    is_new_chat = request_json['new_chat']
    data = {'success': False}

    if is_new_chat or user_id not in client_info.keys():
        clear_user(user_id)

    idx = client_info[user_id]['idx']
    curr_req = client_info[user_id]['curr_req']

    if curr_req is not None:
        if text == '1':
            answer = df_2.loc[df_2['Question'] == curr_req, 'Resp'].iloc[0] + '\n\n' + next_query()

            client_info[user_id]['curr_req'] = None
            client_info[user_id]['idx'] = None

        elif text == '0':
            answer = get_req_list(idx)
            client_info[user_id]['curr_req'] = None

        else:
            answer = 'Введите 1 для ознакомления с ответственностью за невыполнение данного требования. Либо ' \
                     'введите 0 для возврата на предыдущий шаг.'

        data['success'] = True

    elif text.isdigit() and idx is not None:
        if int(text) > df_1['n_answers'].iloc[idx] or text == '0':
            answer = 'Данный индекс не входит в диапазон.'

        else:
            try:
                requirement = re.findall(f'({text}\) )(.*)\n\n\n', df_1.iloc[idx]['Answer'])[0][1]
                answer = 'НПА про ' + requirement.lower()[:-1] + ':' + '\n\n' + df_2.loc[df_2['Question'] ==
                                                                                    requirement, 'Answer'].iloc[0] + \
                         '\n\n' + 'Могу рассказать об ответственности за невыполнение данного требования. Для ' \
                                  'этого введите 1. Либо введите 0 для возврата на предыдущий шаг.'

                client_info[user_id]['curr_req'] = requirement

            except IndexError:
                answer = 'Что-то пошло не так. Попробуйте ещё раз.'

        data['success'] = True

    else:
        not_about_resp = lr_model.predict([text])
        sentence = embed_sentence(text)

        if not_about_resp:
            index, distance = annoy_index.get_nns_by_vector(sentence, 1, include_distances=True)
            if distance[0] > 0.68:
                answer = no_answer()
            else:
                idx = index[0]
                client_info[user_id]['idx'] = idx
                answer = get_req_list(idx)
                data['success'] = True

        else:
            index, distance = annoy_index_resp.get_nns_by_vector(sentence, 1, include_distances=True)
            if distance[0] > 0.68:
                answer = no_answer()
            else:
                answer = df_2['Resp'].iloc[index[0]] + '\n\n' + next_query()
                data['success'] = True

    data['answer'] = answer

    return jsonify(data)


def clear_user(user_id: str) -> None:
    global client_info
    client_info[user_id] = {'idx': None,
                            'curr_req': None}


def embed_sentence(sentence):
    text = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt').to(model.device)

    with torch.inference_mode():
        model_output = model(**text)

    token_embeddings = model_output[0]
    expanded_mask = text['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()

    return (torch.sum(token_embeddings * expanded_mask, dim=1) /
            torch.clamp(expanded_mask.sum(dim=1), min=1e-9)).squeeze()


def get_req_list(idx: int) -> str:
    return df_1['Answer'].iloc[idx] + f'Могу рассказать о нормативно-правовых актах по любому из ' \
                                      f'этих требований. Введите число от 1 до ' \
                                      f'{df_1["n_answers"].iloc[idx]} для выбора требования. Либо ' \
                                      f'введите новый запрос.'


def next_query():
    return 'Пожалуйста, введите следующий запрос.'


def no_answer():
    return 'К сожалению, у меня нет ответа на этот вопрос. Пожалуйста, постарайтесь переформулировать ' \
           'его или запишитесь на консультацию.'


if __name__ == '__main__':
    df_1 = pd.read_csv('./qa_dataset_1.csv')
    df_2 = pd.read_csv('./qa_dataset.csv')
    model_name = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModel.from_pretrained(model_name).to(device)
    annoy_index = annoy.AnnoyIndex(model.config.hidden_size, 'angular')
    annoy_index.load('./models/qa_annoy_model_1.ann')
    annoy_index_resp = annoy.AnnoyIndex(model.config.hidden_size, 'angular')
    annoy_index_resp.load('./models/qa_annoy_model_resps.ann')
    with open('./models/lr_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    client_info = {}

    app.run(host='0.0.0.0', port=8080)
