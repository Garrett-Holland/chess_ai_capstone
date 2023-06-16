# libraries
import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
import io
import time
import random
import chess.pgn
import chess.svg
import chess
from chess import Board, SQUARES, Move
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from IPython.display import display, HTML
from stockfish import Stockfish
from scipy import stats


# Load the data and models
df = pd.read_csv('df.csv')

# Inside your Streamlit app section
pgn_file_path = "../data/all_games.pgn"

# Read the first 15 lines from the PGN file
with open(pgn_file_path, "r") as file:
    pgn_lines = []
    for i, line in enumerate(file):
        pgn_lines.append(line.strip())
        if i == 14:  # Stop after reading 15 lines
            break

model = tf.keras.models.load_model('full_model.h5')
model2 = tf.keras.models.load_model('early_game_model.h5')

# Load the test data
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
X_test2 = np.load('X_test2.npy')
y_test2 = np.load('y_test2.npy')

# Get model summary
def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string

model_summary = get_model_summary(model)
model2_summary = get_model_summary(model2)

# Load the saved model metrics
model_metrics = np.load('model_metrics.npy', allow_pickle=True)
model_metrics2 = np.load('model2_metrics.npy', allow_pickle=True)

model_names = ["First 20 Moves", "First 40 Moves", "First 60 Moves", "All Moves"]
accuracy_scores = [0.28, 0.12, 0.08, 0.05]

# Initialize Stockfish engine
stockfish = Stockfish(path="..\stockfish\stockfish-windows-2022-x86-64-avx2.exe")
stockfish.update_engine_parameters({"Hash": 2048, "UCI_Chess960": "true"})
stockfish.set_skill_level(5)
stockfish.set_elo_rating(500)

# Define a function to convert a board object to a one-hot encoded representation
def board_to_tensor(board):
    piece_mapping = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    tensor = np.zeros((8, 8, 12), dtype=np.float32)
    for square in chess.SQUARES:
        piece_type = board.piece_type_at(square)
        if piece_type is not None:
            piece_index = piece_mapping[chess.piece_symbol(piece_type)]
            row = square // 8
            col = square % 8
            tensor[row, col, piece_index] = 1
    return tensor

def main():
    # Define the sections and their corresponding headers
    sections = {
        "Introduction": "Welcome to my Chess AI Capstone Project!",
        "Data Overview": "Data Overview",
        "Limitations": "Limitations",
        "Stockfish": "Stockfish",
        "Chess AI Model": "Chess AI Model",
        "Model Performance": "Results",
        "Simulation": "Simulation",
        "Conclusion": "Conclusion",
        "Future Integrations": "Future Integrations",
        "Sources": "Sources"
    }

    # Create the sidebar menu
    st.sidebar.title("Menu")
    selected_section = st.sidebar.radio("Go to", list(sections.keys()))

    # Display the selected section
    st.title(sections[selected_section])
    if selected_section == "Introduction":
        introduction_text = """
        <h4>Is it possible to create an autonomous model that is capable of playing chess?</h4>
        <h4>If so, how accurate will it be?</h4>
        <br><br>
        <h4>These are the questions that my project aims to answer.</h4>
        <br><br>
        <h4>Outline:</h4>
        <ul>
            <li>Take a large amount of grandmaster gameplay from top chess players around the world.</li>
            <li>Convert all of that data into board states.</li>
            <li>Use a chess engine called Stockfish to analyze all board states and predict the best possible move for each one.</li>
            <li>Feed all of this data into my model and train it using a Feed Forward Neural Network.</li>
            <li>Conduct analysis on the trained model.</li>
            <li>Create an interactive simulation to run my model against Stockfish.</li>
        </ul>
        <br><br>
        In this app, you can explore detailed chess data, various machine learning techniques, and even run your own simulation against a chess engine.
        """
        st.markdown(introduction_text, unsafe_allow_html=True)
    elif selected_section == "Data Overview":
        data_overview_text = """
        <h4>The dataset used for training the model consists of game records from various top chess players around the world.</h4>
        <br><br>
        <h4>I have imported, processed, and formatted roughly 30,000 games along with 2.5 million board states to train the model on.</h4>
        <br><br>
        <h4>Here is an example of the raw data before processing:</h4>
        Raw PGN Data:
        """
        for line in pgn_lines:
            data_overview_text += line + "<br>"
        data_overview_text += """
        <br><br>
        <h4>Here is the data after processing and formatting:</h4>
        """
        st.markdown(data_overview_text, unsafe_allow_html=True)
        st.dataframe(df)
    elif selected_section == "Limitations":
        limitations_text = """
        <h4>Even though I am training it on 2.5 million board states, this only covers a fraction of a percent of all possible board states.</h4>
        """
        st.markdown(limitations_text, unsafe_allow_html=True)
    elif selected_section == "Stockfish":
        stockfish_text = """
        <h4>Stockfish is a library that had a massive impact on my project.</h4>
        <br><br>
        <h4>Using Stockfish, I was able to calculate the best possible move given a particular board state and feed that information into my model.</h4>
        <br><br>
        <h4>Stockfish also has a chess engine that you can play against while adjusting the skill level to your liking.</h4>
        <br><br>
        <h4>This allowed me to create a simulation for my model and test its abilities.</h4>
        """
        st.markdown(stockfish_text, unsafe_allow_html=True)
    elif selected_section == "Chess AI Model":
        chess_model_text = """
        <h4>The model is built using a Feed Forward Neural Network.</h4>
        <h4>It takes the chess board state as input and predicts the best move to make.</h4>
        <h4>It is trained on board states from top players around the world as well as the best move recommended by Stockfish.</h4>
        """
        st.markdown(chess_model_text, unsafe_allow_html=True)
        st.text(model_summary)
    elif selected_section == "Model Performance":
        model_performance_text = """
        <h4>My Model agrees with stockfish roughly 5% of the time.</h4>
        <h4>This doesn't mean that 95% of the time the model is wrong, just that it agrees with stockfish's exact move 5% of the time.</h4>
        <br>
        <h4>After seeing this, I wanted to test the model's performance throughout each game, so I plotted the model's accuracy in 20 move increments.</h4>
        """
        st.markdown(model_performance_text, unsafe_allow_html=True)
        # Create a bar chart of the accuracy scores
        plt.bar(model_names, accuracy_scores)
        plt.xlabel("Models")
        plt.ylabel("Accuracy Scores")
        plt.title("Accuracy Scores of Models")
        plt.ylim(0, 1)  # Set the y-axis limits to range from 0 to 1

        # Display the chart in Streamlit
        st.pyplot(plt)
        st.markdown("<br><br>", unsafe_allow_html=True)
        performance_summary_text = "<h4>As you can see, the model performs extremely well in the early game but begins to fall off as the game progresses.</h4>"
        st.markdown(performance_summary_text, unsafe_allow_html=True)
    elif selected_section == "Simulation":
        # Create a slider to adjust the Stockfish skill level
        skill_level = st.slider("Stockfish Skill Level", min_value=1, max_value=20, value=5)
    
        # Inside your Streamlit app section
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        board_svg = chess.svg.board(chess.Board(fen), size=350)
        board_container = st.markdown(board_svg, unsafe_allow_html=True)

        # Define a function to update the board SVG
        def update_board_svg(fen):
            new_board_svg = chess.svg.board(chess.Board(fen), size=350)
            board_container.markdown(new_board_svg, unsafe_allow_html=True)

        # Define a function to convert a board object to a one-hot encoded representation
        def board_to_tensor(board):
            piece_mapping = {
                'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
            }
            tensor = np.zeros((8, 8, 12), dtype=np.float32)
            for square in chess.SQUARES:
                piece_type = board.piece_type_at(square)
                if piece_type is not None:
                    piece_index = piece_mapping[chess.piece_symbol(piece_type)]
                    row = square // 8
                    col = square % 8
                    tensor[row, col, piece_index] = 1
            return tensor
        
        legal_moves = []

        for a in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
            for b in ['1', '2', '3', '4', '5', '6', '7', '8']:
                for c in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
                    for d in ['1', '2', '3', '4', '5', '6', '7', '8']:
                        legal_moves.append(a + b + c + d)
                        if d == '1' or d == '8':
                            legal_moves.append(a + b + c + d + 'r')
                            legal_moves.append(a + b + c + d + 'q')
                            legal_moves.append(a + b + c + d + 'n')
                            legal_moves.append(a + b + c + d + 'b')

        # Create a label encoder object
        label_encoder = LabelEncoder()

        label_encoder.fit(legal_moves)

        # Main game loop
        while True:
            # My model's turn
            board_state = chess.Board(fen)
            board_state_array = np.array(board_to_tensor(board_state)).reshape(1, 8, 8, 12)
            prediction = model.predict(board_state_array)

            move = list(board_state.legal_moves)
            moves = [str(move) for move in list(board_state.legal_moves)]
            move_indeces = label_encoder.transform(moves)

            best_move = random.choices(moves, weights=prediction[0][move_indeces])[0]

            st.write('AI move:', best_move)

            # Update the board state
            board_state.push_san(best_move)
            fen = board_state.fen()

            # Update the board SVG
            update_board_svg(fen)

            # Introduce a delay of 1 second
            time.sleep(1)

            stockfish.set_fen_position(fen)
            stockfish.set_skill_level(skill_level)

            # Check if the game is over
            if board_state.is_game_over():
                last_move = 'AI'
                break

            # Stockfish's turn
            stockfish_best_move = stockfish.get_best_move()

            st.write('Stockfish move:', stockfish_best_move)

            # Update the board state
            board_state.push_san(stockfish_best_move)
            fen = board_state.fen()

            # Update the board SVG
            update_board_svg(fen)

            # Introduce a delay of 1 second
            time.sleep(1)

            stockfish.set_fen_position(fen)

            # Check if the game is over
            if board_state.is_game_over():
                last_move = 'Stockfish'
                break

        if board_state.is_checkmate():
            st.write('Winner:', last_move)
        else:
            st.write('Stalemate')
    elif selected_section == "Conclusion":
        conclusion_text = """
        <h4>This is V1 of the model. I will constantly be updating and enhancing the model with methods of improvement.</h4>
        <h4>As of now, the model is performing extremely well in the early game and falling off the further into the game it gets.</h4>
        """
        st.markdown(conclusion_text, unsafe_allow_html=True)
    elif selected_section == "Future Integrations":
        future_integrations_text = """
        <h4>I plan to integrate a capture system on the simulated games so it can learn outside of grandmaster gameplay.</h4>
        <h4>I plan to experiment with other machine learning methods to improve late game results.</h4>
        <h4>Add weights to the pieces so that it prioritizes important pieces.</h4>
        """
        st.markdown(future_integrations_text, unsafe_allow_html=True)
    elif selected_section == "Sources":
        sources_text = """
        <h4>chess.com</h4>
        <p>- Used to pull in all of the raw data from top chess players around the world.</p>
        <h4>chess library</h4>
        <p>- Used for formatting purposes.</p>
        <h4>stockfish library</h4>
        <p>- Used to calculate the best move.</p>
        <p>- Allowed me to test my model against a chess engine.</p>
        """
        st.markdown(sources_text, unsafe_allow_html=True)
if __name__ == "__main__":
    main()