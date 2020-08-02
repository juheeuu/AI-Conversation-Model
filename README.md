# AI conversation Model with User-MMI

### Abstract 
- To generate engaging conversation responses, we have to utilize speaker(user) information at training time. Because depending on user information, the answer can be different despite of the same question. 
- The previous approach suffers problems that generating response with inconsistent peronality or generating response with average personality of all users. 

![](https://i.imgur.com/oEHlKeX.png)

- To handle these problem and generate more probable response based on the user information, I suggest the **User-MMI**

### User MMI 
- T: target sentence, S: source sentence, M: meta inforamtion
- This is the math of the User-MMI
![](https://i.imgur.com/asbe37K.png)
- Model example
![](https://i.imgur.com/8qcHOwr.png)


## Experiment

### Quantitative experiment

- Experiment result of Cornell Movie Dataset 

| Model    | BLEU     | Embedding | METEOR | R-L Precision | R-L Recall | R-L F1 | 
| -------- | -------- | --------  |--------|    --------   |  --------  |--------|
| DialoGPT | 0.1435 | 0.2224    |0.0946|0.0686|0.0890|0.0637|
| DialoGPT + MMI |0.1453 | 0.2252 |0.0963|0.0696|0.0893|0.0645|
| DialoGPT + User |0.1516| 0.7452 |0.0905|0.0905|0.0886|0.0724|
| DialoGPT + User MMI|**0.1553**| **0.7487** |**0.0943**|**0.0943**|**0.0890**|**0.0742**|

- Experiment result of Reddit Dataset 
    1. Crawwled reddit dataset to check generalization ability of the conversation model. 
    2. 146519 over 20766 dialogs 12277 speakers 
    3. New Speaker Number for validation & Test 956 / 6492

| Model    | BLEU     | Embedding | R-L Precision | R-L Recall | R-L F1 | 
| -------- | -------- | --------  |--------|    --------   |  --------  |
| DialoGPT | 0.0877 | 0.8012    |0.0544|0.0665|0.0480|0.0637|
| DialoGPT + MMI |0.0882 | 0.8066 |0.0590|0.0680|0.0496|0.0645|
| DialoGPT + User |0.0956|** 0.8381** |0.0786|**0.0608**|0.0542|
| DialoGPT + User MMI|**0.0970**|0.8207|**0.0873** |0.0585|**0.0548**|

### Qualitative experiment
- Experiment with persona chat data 
- Resolve average personality problem. 
    ![](https://i.imgur.com/X1CFXzR.png)
- Resolve inconsistent personality problem 
![](https://i.imgur.com/daEitEl.png)

