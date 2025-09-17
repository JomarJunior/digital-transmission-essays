from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np


def load_mat_file(file_path: str):
    """Load a .mat file and return its contents."""
    data = loadmat(file_path)
    return data


if __name__ == "__main__":
    mat_data = load_mat_file("config/channel_models/NB_0_500k.mat")
    time_response = np.array(mat_data["h"])
    time_response = (
        time_response.ravel()
    )  # this is to convert from 2D to 1D array if needed
    print(time_response)
    print("Channel Impulse Response Length:", len(time_response))
    frequency_response = np.fft.fft(time_response, n=2 * 64)  # Assuming 64 subcarriers

    # Plot both impulse response and frequency response
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.stem(time_response)
    plt.title("Channel Impulse Response")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(frequency_response)
    plt.title("Channel Frequency Response Magnitude")
    plt.xlabel("Frequency Bin")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.tight_layout()
    plt.savefig("channel_response.png", dpi=150)
    plt.show()
