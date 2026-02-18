"""
This file stores instructions for various question types in a dictionary format, providing a structured and easily accessible reference for downstream processing.
The dictionary maps question categories to their corresponding instructions, enabling efficient retrieval and usage across the application.
"""

peak_number_CH_instruction = "下面会输入一张图像和一个问题，你的最终回答应该是一个数字，不要有多余的输出。如果问题无法回答或你无法识别到任何峰，请回答“None”。"
peak_position_CH_instruction = "下面会输入一张图像和一个问题，请根据问题先识别并分析图中X轴的坐标及刻度信息，再给出峰的横坐标（X轴坐标）。如果你从图像中识别到了多个峰，请从图像的左至右，分别回答各个峰的横坐标（X轴坐标）。你的最终回答应该是一个或多个数字，多个数字之间用“;”分隔（例如：520;560;580），不要有多余的输出。如果问题无法回答或你无法识别到任何峰，请回答“None”。"
peak_value_CH_instruction = "下面会输入一张图像和一个问题，请根据问题先识别并分析图中Y轴的坐标及刻度信息，再给出峰的纵坐标。如果你从图像中识别到了多个峰，请从图像的左至右，分别回答各个峰的纵坐标（Y轴坐标）。你的最终回答应该是一个或多个数字，多个数字之间用“;”分隔（例如：12000;15000;8000），不要有多余的输出。如果问题无法回答或你无法识别到任何峰，请回答“None”。"
fwhm_CH_instruction = "下面会输入一张图像和一个问题，如果你从图像中识别到了多个峰，请从图像的左至右，分别回答各个峰的半峰宽。你的最终回答应该是一个或多个数字，多个数字之间用“;”分隔（例如：20;30;7），不要有多余的输出。如果问题无法回答或你无法识别到任何峰，请回答“None”。计算半峰宽的方法如下：1.找到峰值的最大高度，即峰的最高点。2.计算该高度的一半，即峰值高度的一半。3.在图中找到与这个一半高度相对应的两个位置，一个在峰的左侧，一个在右侧。4.计算这两个位置之间的横坐标距离，该距离就是半峰宽。"
shape_CH_instrcution = "下面会输入一张图像和一个问题，如果输入的文本是英语，你的最终回答应该是“yes”或“no”。如果输入的文本是汉语，你的最终回答应该是“是”或“否”。不要有多余的输出。"
Rea_CH_instrucion = "下面会输入一张图像和一个问题，图像由多个图表组合而成，如果问题类型是选择题，每个图表下方对应着一个字母作为选项。你的最终回答应该是一个字母（如“A”）。如果问题类型是判断题，如果输入的文本是英语，你的最终回答应该是“yes”或“no”，如果输入的文本是汉语，你的最终回答应该是“是”或“否”。注意不要有多余的输出。"

peak_number_EN_instruction = "An image and a question will be entered below, and your final answer should be a number with no extra output. If the question is unanswerable or you can't recognize any peaks, answer “None”."
peak_position_EN_instrucion = "An image and a question will be entered following, please identify and analyze the X-axis coordinates and scale information in the image before giving the horizontal coordinates (X-axis coordinates) of the peaks according to the question. If you recognize more than one peak from the image, answer the horizontal coordinates (X-axis coordinates) of each peak from left to right of the image. Your final answer should be one or more numbers, with multiple numbers separated by a “;” (e.g., 520;560;580), with no extra output. If the question is unanswerable or you can't recognize any peaks, answer “None”."
peak_value_EN_instrucion = "An image and a question will be entered following, please identify and analyze the coordinates and scale information of the Y-axis in the image before giving the vertical coordinates of the peaks according to the question. If you recognize more than one peak from the image, please answer the vertical coordinates (Y-axis coordinates) of each peak from left to right of the image. Your final answer should be one or more numbers, with “;” separating multiple numbers (e.g., 12000;15000;8000) and no extra output. If the question is unanswerable or you can't recognize any peaks, answer “None”."
fwhm_EN_instrucion = "An image and a question will be entered below, if you identify more than one peak from the image, please answer the full width at half maximum of each peak from left to right of the image. Your final answer should be one or more numbers, with multiple numbers separated by a “;” (e.g., 20;30;7), with no extra output. If the question is unanswerable or you can't recognize any peaks, answer “None”. The method to calculate the Full Width at Half Maximum (FWHM) is as follows: 1. Identify the peak's maximum height, which is the highest point of the peak. 2. Calculate half of this height, which is half of the peak's maximum height. 3. Find the two positions on the graph corresponding to this half height, one on the left side of the peak and one on the right side. 4. Calculate the horizontal distance between these two positions, which is the Full Width at Half Maximum."
shape_EN_instrucion = "An image and a question will be entered below. If the input text is in English, your final answer should be ‘yes’ or ‘no’. If the input text is in Chinese, your final answer should be ‘是’或‘否’. Note that there is no extra output."
Rea_EN_instrucion = "An image and a question will be entered below, with the image consisting of a combination of multiple charts. If the question type is multiple choice, each chart has a letter below it as an option. Your final answer should be a letter (e.g. “A”). If it is a True or False question, your final answer should be “yes” or “no” if the input text is in English, or “是” or “否” if the input text is in Chinese. Note that there is no extra output."

instruction = {
    "peak_number_CH":peak_number_CH_instruction,
    "peak_position_CH":peak_position_CH_instruction,
    "peak_value_CH":peak_value_CH_instruction,
    "fwhm_CH":fwhm_CH_instruction,
    "shape_CH":shape_CH_instrcution,
    "Rea_CH":Rea_CH_instrucion,
    "peak_number_EN":peak_number_EN_instruction,
    "peak_position_EN":peak_position_EN_instrucion,
    "peak_value_EN":peak_value_EN_instrucion,
    "fwhm_EN":fwhm_EN_instrucion,
    "shape_EN":shape_EN_instrucion,
    "Rea_EN":Rea_EN_instrucion,
    }
