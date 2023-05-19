<<<<<<< HEAD

=======
<!--
 * @Author: lihaitao
 * @Date: 2023-05-16 19:48:12
 * @LastEditors: Do not edit
 * @LastEditTime: 2023-05-16 20:44:08
 * @FilePath: /lht/LexiLaw/README.md
-->
>>>>>>> a70d566b85ef28f2eee2004766e1105ade5d1d43
# LexiLaw - 中文法律大模型

欢迎来到 LexiLaw 项目！这个项目旨在提供专业的中文法律咨询服务，并分享在大模型在垂直领域微调的经验，以帮助社区开发更多优质的专用领域的大模型。

## 项目简介

LexiLaw 是一个经过微调的中文法律大模型，它基于 ChatGLM-6B 架构，通过在法律领域的数据集上进行微调，使其在提供法律咨询和支持方面具备更高的性能和专业性。

该模型旨在为法律从业者、学生和普通用户提供准确、可靠的法律咨询服务。无论您是需要针对具体法律问题的咨询，还是对法律条款、案例解析、法规解读等方面的查询，LexiLaw 都能够为您提供有益的建议和指导。

同时，我们将分享在大模型基础上微调的经验和最佳实践，以帮助社区开发更多优秀的中文法律大模型，推动中文法律智能化的发展。

## 功能和特点

- **专业法律知识**：LexiLaw 经过在大规模法律数据集上的微调，拥有丰富的中文法律知识和理解能力，能够回答各类法律问题。

- **法律咨询服务**：通过与 LexiLaw 进行交互，您可以提出具体的法律问题，模型将根据您的输入提供详细和准确的回答，为您提供法律咨询和支持。

- **广泛应用场景**：LexiLaw 适用于各种法律领域，包括但不限于合同法、劳动法、知识产权、民事诉讼、刑事法等。无论您是法律从业者、学生还是需要法律帮助的个人，我们都希望通过这个模型为您提供有价值的支持。

- **经验分享**：我们将分享在大模型微调方面的经验和最佳实践，帮助社区的开发者们更好地构建和微调中文法律大模型，推动法律智能化的进步。

- **持续更新**：我们会不断更新和改进模型，以确保它与最新的法律发展和变化保持同步。您可以定期获取最新的模型版本和更新内容。

## 如何使用

1. 克隆或下载本项目到您的本地环境。
<<<<<<< HEAD

2. 安装所需的依赖项和配置环境（请参考项目文档中的说明）。

3. 运行 LexiLaw 模型或将其集成到您的应用程序中。
=======
    ```
    git clone https://github.com/CSHaitao/LexiLaw.git
    cd LexiLaw
    ```

2. 安装所需的依赖项和配置环境。

    ```
    pip install -r requirements.txt
    ```

3. 下载训练好的参数[LoRA] [P-tuningv2] [Finetune]。
>>>>>>> a70d566b85ef28f2eee2004766e1105ade5d1d43

4. 通过与模型进行交互，提供具体的法律问题或相关法律文本，LexiLaw 将根据您的输入提供相应的回答和解释。

## 训练数据

LexiLaw 的训练数据是通过综合使用通用领域数据、专业法律数据和法律文书进行微调而得到的。我们发现仅使用法律领域数据进行微调容易导致严重的过拟合现象，会导致模型忘掉原有的能力。

因此, 我们采用了以下数据组合来丰富模型的知识和能力：

<<<<<<< HEAD
- **通用领域数据**：我们使用了大规模的通用领域文本数据集 [BELLE](https://github.com/LianjiaTech/BELLE) 1.5M，其中包括不同指令类型、不同领域的文本。通过引入通用领域数据，模型可以更好地理解自然语言和上下文信息，提高对各种问题的处理能力。
=======
- **通用领域数据**：我们使用了大规模的通用领域文本数据集 **[BELLE](https://github.com/LianjiaTech/BELLE)** 1.5M，其中采样了30k。其中包括不同指令类型、不同领域的文本。通过引入通用领域数据，模型可以更好地理解自然语言和上下文信息，提高对各种问题的处理能力。
>>>>>>> a70d566b85ef28f2eee2004766e1105ade5d1d43

- **法律问答数据**：我们收集了丰富的法律问答数据，包括常见法律问题和相应的答案。这些问答数据涵盖了多个法律领域，如合同法、劳动法、知识产权等。通过使用这些数据，模型可以学习到常见法律问题的回答模式和解释能力。

    - **[LawGPT_zh](https://github.com/LiuHC0428/LAW-GPT)** :52k单轮问答数据和92k带有法律依据的情景问答
    - **[Lawyer LLaMA](https://github.com/AndrewZhe/lawyer-llama)** :法考数据和法律指令微调数据
    - **[华律网问答数据](https://www.66law.cn/)** :20k高质量华律网问答数据
    - **[法律知道](https://github.com/murufeng/ChineseNlpCorpus/blob/master/datasets/lawzhidao/intro.ipynb)** :百度知道收集的36k条法律问答数据

- **法律法规和法律参考书籍**：为了增强模型对法律条款和法规的理解，我们引入了包括法条和法律参考书籍在内的专业法律文本。这些文本提供了具体的法律规定和解释，帮助模型学习并提供更准确的法律解释和建议。

    - **[法律法规](https://github.com/LawRefBook/Laws)** 包含刑法、民法、宪法、司法解释等法律法规。构建以下格式的训练数据:

        > {"instruction": "给出以下法条的具体内容:", "input": " 畜禽规模养殖污染防治条例第二条", "answer": " 畜禽规模养殖污染防治条例第二条 本条例适用于畜禽养殖场、养殖小区的养殖污染防治。畜禽养殖场、养殖小区的规模标准根据畜牧业发展状况和畜禽养殖污染防治要求确定。牧区放牧养殖污染防治，不适用本条例。"}

    - **[法律参考书籍](https://jecqa.thunlp.org/)** JEC-QA数据集提供的法律参考书籍，可以通过 [JEC-QA](https://jecqa.thunlp.org/) 官网申请获得参考数据。构建以下格式的训练数据:

        > {"instruction": "请解释以下概念的具体含义:", "input": "管辖恒定", "answer": "管辖恒定，是指确定案件的管辖权，以起诉时为标准:起诉时对案件享有管辖权的法院，不因确定管辖的相关因素在诉讼过程中发生变化而影响其管辖权。\",管辖恒定包括级别管辖恒定和地域管辖恒定。《民诉解释》第37条规定:“案件受理后，受诉人民法院的管辖权不受当事人住所地.经常居住地变更的影响。”第38条规定:“有管辖权的人民法院受理案件后，不得以行政区域变更为由，将案件移送给变更后有管辖权的人民法院。判决后的上诉案件和依审判监督程序提审的案件，由原审人民法院的上级人民法院进行审判；上级人民法院指令再审.发回重审的案件，由原审人民法院再审或者重审。”第39条规定:“人民法院对管辖异议审查后确定有管辖权的，不因当事人提起反诉.增加或者变更诉讼请求等改变管辖，但违反级别管辖.专属管辖规定的除外。人民法院发回重审或者按第一审程序再审的案件，当事人提出管辖异议的，人民法院不予审查。”\","}


- **法律文书**：为了让模型熟悉实际的法律案例和文书，我们还使用了包括判决书、法院裁定书和法律文件等在内的法律文书。这些文书涵盖了各种法律领域和案件类型，使模型能够学习到真实案例的背景、判断和解决方法。

    - **[法律文书](https://wenshu.court.gov.cn/)** 从法律裁判文书网收集50k法律文书，使用正则匹配提取文书的事实部分和裁判分析过程部分，构建以下格式的训练数据:

        > {"instruction": "你现在是一个精通中国法律的法官，请对以下案件做出分析:", "input": "经审理查明：2015年5月，被告人胡兴旺通过王某2以帮助被害人王某1“捞人”为由，在本市丰台区丰益桥工商银行门前等地，诈骗被害人王某1人民币30余万元。被告人胡兴旺于2016年9月20日被北京市公安局丰台分局刑侦支队抓获。案发前胡兴旺向王某2归还人民币12.8万元，案发后向王某2归还800元；案发后向被害人王某1归还人民币28万余元。", "answer": "本院认为，被告人胡兴旺以非法占有为目的，虚构事实，骗取他人财物，数额巨大，其行为已构成诈骗罪，应予处罚。北京市丰台区人民检察院指控被告人胡兴旺犯诈骗罪的主要事实清楚、证据确实充分，罪名成立，唯被告人在案发前归还的部分钱款应从犯罪金额中扣除。鉴于被告人胡兴旺到案后能够如实供述犯罪事实，认罪态度较好，且已退还被害人经济损失，故本院依法对其予以从轻处罚并宣告缓刑。被告人胡兴旺的辩解及辩护人的辩护意见本院酌予采纳。"}

通过综合利用通用领域数据、专业法律数据、法律问答数据和法律文书，LexiLaw 在处理法律问题时能够提供准确、全面的法律咨询和支持。

请注意，LexiLaw 的训练数据仅用于模型微调和改进，不包含个人敏感信息或侵犯隐私的内容。

<<<<<<< HEAD

## 贡献和改进


=======
## 模型训练

我们采用以下三种方式对 ChatGLM-6B 进行了深度微调,所有的模型都是在 7 张40G A100上训练模型，训练代码使用DeepSpeed和Trainer，具体说明可见[ChatGLM_mutli_gpu_tuning](https://github.com/CSHaitao/ChatGLM_mutli_gpu_tuning).

1. **LoRA**
    运行`sh lora.sh`  
    具体参数如下：
    ```bash
    CUDA_VISIBLE_DEVICES=${TOT_CUDA} deepspeed --master_port=$PORT --num_gpus=7 lora.py \
        --train_path ./instrution_data.json \
        --max_len 768 \
        --max_input_len 512 \
        --model_name_or_path ./chatGLM-6B \
        --tokenizer_name ./chatGLM-6B \
        --lora_rank 32 \
        --per_device_train_batch_size 12 \
        --gradient_accumulation_steps 2 \
        --num_train_epochs 6 \
        --save_strategy epoch \
        --learning_rate 5e-4 \
        --fp16 \
        --remove_unused_columns false \
        --logging_steps 50 \
        --output_dir /output \
        --deepspeed /ds_config.json \
    ```
    lora_rank = 32，训练参数量情况：
    ```
    trainable params: 14680064 || all params: 6187966464 || trainable%: 0.23723567484414862 
    ```

2. **P-tuning-v2**
    运行 `sh ptuning.sh`
    具体参数如下：
    ```bash
        CUDA_VISIBLE_DEVICES=${TOT_CUDA} deepspeed --master_port=$PORT --num_gpus=2 finetune_ptuning.py \
            --train_path ./instrution_data.json \
            --max_len 768 \
            --max_input_len 512 \
            --model_name_or_path /chatGLM-6B \
            --tokenizer_name/chatGLM-6B \
            --per_device_train_batch_size 8 \
            --gradient_accumulation_steps 4 \
            --num_train_epochs 10 \
            --save_strategy epoch \
            --learning_rate 5e-4 \
            --fp16 \
            --logging_steps 50 \
            --pre_seq_len 128 \
            --output_dir /output \
            --deepspeed ds_config.json \
    ```
    pre_seq_len = 128，prefix_projection = True，训练参数量情况：
    ```
    trainable params: 957059072 || all params: 7130345472 || trainable%: 13.42233803058007
    ```
    
3. **Finetune**
    运行`sh freeze.sh`
    具体参数如下：
    ```bash
    CUDA_VISIBLE_DEVICES=${TOT_CUDA} deepspeed --master_port=$PORT --num_gpus=3 finetune_freeze.py \
        --train_path  \
        --max_len 768 \
        --max_input_len 512 \
        --model_name_or_path /chatGLM-6B \
        --tokenizer_name /chatGLM-6B \
        --per_device_train_batch_size 16 \
        --gradient_accumulation_steps 4 \
        --num_train_epochs 4 \
        --save_strategy epoch \
        --learning_rate 1e-5 \
        --fp16 \
        --remove_unused_columns false \
        --logging_steps 50 \
        --output_dir output_freeze \
        --deepspeed ds_config.json \
    ```
    `finetune_freeze.py` 中设置只训练 `layers.27,layers.26,layers.25,layers.24,layers.23`。训练参数量情况：
    ```
    trainable params: 1006899200 || all params: 6173286400 || trainable%: 16.31058620575258
    ```









## 贡献和改进

>>>>>>> a70d566b85ef28f2eee2004766e1105ade5d1d43
贡献和改进是推动 LexiLaw 项目持续发展的重要因素。您可以通过以下方式参与和支持项目：

- **问题和反馈**：如果您在使用 LexiLaw 时发现任何问题、错误或有改进建议，请在 GitHub 的 Issues 页面上提出您的反馈。我们欢迎任何关于性能、功能或用户体验方面的问题和意见。

- **Pull 请求**：如果您有能够改进 LexiLaw 的代码、功能或文档的想法，欢迎提交 Pull 请求。我们将仔细审查您的贡献，并与您合作以确保项目的质量和稳定性。

- **分享经验**：如果您在微调大模型方面有独特的经验和最佳实践，我们鼓励您将其分享给社区。您可以编写博客文章、示例代码或提供文档来帮助其他开发者更好地理解和使用 LexiLaw。

## 注意事项

- LexiLaw 是基于深度学习技术构建的，它可以提供有价值的法律建议和解释，但不应视为法律专家的替代品。在重要的法律事务中，建议您咨询专业的法律顾问或律师。

- 本项目遵循适用的开源许可证。请在使用或分发代码之前，详细阅读项目中的许可证文件。

<<<<<<< HEAD
=======
## 致谢

本项目参考了以下开源项目，在此对相关项目和研究开发人员表示感谢。

- LawGPT_zh：https://github.com/LiuHC0428/LAW-GPT
- Lawyer LLaMA：https://github.com/AndrewZhe/lawyer-llama
- Laws： https://github.com/LawRefBook/Laws
- ChineseNlpCorpus：https://github.com/murufeng/ChineseNlpCorpus
- LuXun-GPT：https://github.com/Suffoquer-fang/LuXun-GPT

>>>>>>> a70d566b85ef28f2eee2004766e1105ade5d1d43
## 参与讨论

如果您对 LexiLaw 有任何疑问、建议或想法，欢迎加入我们的讨论。您可以联系 liht22@mails.tsinghua.edu.cn 提出问题、参与技术讨论或分享您的见解。

我们衷心感谢您对 LexiLaw 项目的关注和参与！希望通过这个项目，能够为中文法律领域提供更智能、可靠的解决方案。
