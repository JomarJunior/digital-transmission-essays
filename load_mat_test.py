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
    time_response = time_response.ravel()  # this is to convert from 2D to 1D array if needed

    # Normalize the impulse response
    time_response = time_response / np.linalg.norm(time_response, ord=2)

    print(time_response)
    print("Channel Impulse Response Length:", len(time_response))
    frequency_response = np.fft.fft(time_response, n=64, norm="ortho")  # Assuming 64 subcarriers

    # Parseval theorem check
    energy_time = np.sum(np.abs(time_response) ** 2)
    energy_freq = np.sum(np.abs(frequency_response) ** 2) / len(frequency_response)
    print("Energy in Time Domain:", energy_time)
    print("Energy in Frequency Domain:", energy_freq)

    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RESET = "\033[0m"
    if np.isclose(energy_time, energy_freq):
        print(f"{GREEN}Parseval's theorem holds.{RESET}")
    else:
        print(f"{YELLOW}Parseval's theorem does not hold.{RESET}")

    # Db scale the frequency response
    frequency_response = 20 * np.log10(
        np.abs(frequency_response) + 1e-10
    )  # Adding a small value to avoid log(0)

    # Plot both impulse response and frequency response
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(time_response)
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
