<p  style="text-align: center;">Several parts of this README file have been written by AI</p>

<p  style="text-align: center;">you can test the AI by figuring out which parts were. </p>

<br>

  

# AI Detection Program
This is an AI detection program written in Python using TensorFlow, designed to analyze and classify messages from humans and ChatGPT. The program utilizes a dataset filled with Discord messages and ChatGPT-generated messages to train a machine learning model for identifying AI-generated content.

  ![dont put your mouse here.](https://media.discordapp.net/attachments/1075274955098435628/1100931424531513404/image.png?width=827&height=152)

## Requirements
To run this program, you need the following:
- Python 3.x

- TensorFlow library

- Numpy library

- Pandas library

- Scikit-learn library

You can install the required libraries using pip:
`pip install tensorflow numpy pandas scikit-learn`

## Dataset
The dataset used for training this AI detection program consists of Discord messages from the "E - D G Y" Discord server and ChatGPT-generated messages. It is crucial to have a diverse and representative dataset to ensure the effectiveness of the detection model.

The dataset should be organized in a CSV (Comma-Separated Values) format, with each row representing a message and containing the following columns:

-  `class`: Classification of the string (0 = Human, 1 = AI).
-  `message`: The labeled string of text.

![AI traning dataset example](https://media.discordapp.net/attachments/1075274955098435628/1100933448228683836/image.png?width=1439&height=492)

## Model Training
1. Preprocessing: The dataset should be preprocessed to remove unnecessary characters, clean the text, and convert it into a suitable format for model training. The messages cannot contain commas or quotations due to the CSV file format.

2. Model Training: After modifying the dataset CSV files to your needs, you may then run the `train.py` file and let it run the epochs. After the training is over it will ask you for user input and will then classify the message as AI or human generated.

![yes im hosting these on discord, kill me.](https://media.discordapp.net/attachments/1075274955098435628/1100932893209002086/image.png?width=1062&height=69)

## Usage
To use the AI detection program, follow these steps:

1. Ensure that you have installed all the required libraries mentioned in the "Requirements" section.

2. Obtain or create a suitable dataset containing Discord messages and ChatGPT-generated messages in CSV format. Make sure it includes the necessary columns: `class` and `message`.

3. Run the `train.py` script to train the AI detection model. This will save the trained model to a file.

4. Once the model is trained, you can use it for detecting AI-generated content. Modify the `main.py` script to load the trained model and provide new messages as input. The script will predict whether the input messages is written by a human or ChatGPT.

```python main.py```


## Future Improvements
- Increase dataset diversity: Gather a more diverse dataset with a wider range of human messages and ChatGPT-generated text to improve the model's accuracy and generalization.

- Train it on other languages including programming languages to improve classification.

- Make a headless mode for the AI to allow automation.

- Prevent it from detecting the American constitution as AI.
