import random
import gradio as gr


def predict_preferences(user, k):

    scores = [(item, random.random()) for item in range(num_movies)]
    scores.sort(key=lambda x: x[1], reverse=True)
    topk = scores[:k]
    return [f"{item}: {score:.2f}" for item, score in topk]


def clear_predictions():
    return []


def do_predict(user, k):
    try:
        k = int(k)
    except:
        return ["请输入有效的k值"]
    return predict_preferences(user, k)


def get_topk_from_list(user_id, movie_id_list, topk):
    scores = [(item, random.random()) for item in movie_id_list]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [f"{item}" for item, score in scores[:topk]]


def get_topk_at_score_from_list(user_id, score, movie_id_list, topk):
    scores = [(item, random.random()) for item in movie_id_list]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [f"{item}" for item, score in scores[:topk]]


def get_topk(user_id, score, movie_id_list, topk):
    if score == "无":
        score = None
    else:
        score = int(score)
    topk = int(topk)
    if len(movie_id_list) == 0:
        movie_id_list = [i for i in range(num_movies)]
    if score is not None:
        return get_topk_at_score_from_list(user_id, score, movie_id_list, topk)
    else:
        return get_topk_from_list(user_id, movie_id_list, topk)


if __name__ == "__main__":
    num_users = 6000
    num_movies = 1000

    with gr.Blocks(
        css="""
#panel { background: #f9f9f9; padding: 20px; border-radius: 10px; }
.label { font-size: 16px; font-weight: bold; margin-bottom: 5px; }
    """
    ) as demo:
        gr.Markdown(
            """
# 用户电影评分预测可视化界面

此界面允许选择用户，电影，并预测用户对于选中电影的评分。"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("<div class='label'>选择用户</div>")
                user_textbox = gr.Textbox(placeholder="输入user_id", label="user_id")

                gr.Markdown("<div class='label'>选择电影</div>")
                movie_id_list = gr.Dropdown(
                    label="多选电影以在指定范围内预测评分",
                    choices=list(range(1, 1 + num_movies)),
                    multiselect=True,
                    allow_custom_value=True,  # 允许输入新值
                )

            with gr.Column(scale=2):
                gr.Markdown("<div id='panel'><div class='label'>预测展示区</div>")
                pred_output = gr.Textbox(value="", label=None, interactive=False)
                gr.Markdown("</div>")

                with gr.Row():
                    k_input = gr.Textbox(value="3", label="展示数量 k")
                    tgt_score = gr.Dropdown(
                        choices=["无"] + [str(i) for i in range(1, 1 + 5)],
                        label="预测评分筛选",
                    )
                    predict_button = gr.Button("预测偏好")
                    clear_button = gr.Button("清除预测")

                predict_button.click(
                    fn=get_topk,
                    inputs=[user_textbox, tgt_score, movie_id_list, k_input],
                    outputs=[pred_output],
                )
                clear_button.click(
                    fn=lambda: clear_predictions(), inputs=None, outputs=[pred_output]
                )

        demo.launch()
