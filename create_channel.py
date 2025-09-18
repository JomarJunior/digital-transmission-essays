import numpy as np
import matplotlib.pyplot as plt


def create_channel(impulse_response: np.ndarray, file_path: str, filename: str):
    """Create a channel model based on the given impulse response."""
    # Normalize the impulse response
    impulse_response = impulse_response / np.linalg.norm(impulse_response, ord=2)

    # Save in binary format to handle complex numbers
    np.save(f"{file_path}/{filename}.npy", impulse_response)
    print(f"Channel impulse response saved to {file_path}/{filename}.npy")
    return impulse_response


def load_channel(file_path: str) -> np.ndarray:
    """Load a channel model from a given file path."""
    # Load from binary format to handle complex numbers
    impulse_response = np.load(file_path)
    return impulse_response


if __name__ == "__main__":
    # Channel P1
    p1 = np.array(
        [
            0.7768 + 0.4561j,
            -0.0667 + 0.2840j,
            0.1399 - 0.1592j,
            0.0223 + 0.2410j,
        ]
    )

    # Channel P2
    p2 = np.array(
        [
            -0.3699 - 0.5782j,
            -0.4053 - 0.5750j,
            -0.0834 - 0.0406j,
            0.1587 - 0.0156j,
        ]
    )

    # Take the fft of both channels to verify their frequency response
    freq_response_p1 = np.fft.ifft(p1, n=64, norm="ortho")
    freq_response_p2 = np.fft.ifft(p2, n=64, norm="ortho")

    # Verify Parseval's theorem for both channels
    energy_time_p1 = np.sum(np.abs(p1) ** 2)
    energy_freq_p1 = np.sum(np.abs(freq_response_p1) ** 2)
    energy_time_p2 = np.sum(np.abs(p2) ** 2)
    energy_freq_p2 = np.sum(np.abs(freq_response_p2) ** 2)

    print("Channel P1 Energy in Time Domain:", energy_time_p1)
    print("Channel P1 Energy in Frequency Domain:", energy_freq_p1)
    print("Channel P2 Energy in Time Domain:", energy_time_p2)
    print("Channel P2 Energy in Frequency Domain:", energy_freq_p2)

    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RESET = "\033[0m"
    if np.isclose(energy_time_p1, energy_freq_p1):
        print(f"{GREEN}Parseval's theorem holds for Channel P1.{RESET}")
    else:
        print(f"{YELLOW}Parseval's theorem does not hold for Channel P1.{RESET}")

    if np.isclose(energy_time_p2, energy_freq_p2):
        print(f"{GREEN}Parseval's theorem holds for Channel P2.{RESET}")
    else:
        print(f"{YELLOW}Parseval's theorem does not hold for Channel P2.{RESET}")

    # Plot both impulse responses in frequency domain
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.abs(freq_response_p1))  # dB scale
    plt.title("Frequency Response of Channel P1")
    plt.xlabel("Frequency Bin")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(np.abs(freq_response_p2))  # dB scale
    plt.title("Frequency Response of Channel P2 ")
    plt.xlabel("Frequency Bin")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.tight_layout()
    plt.savefig("channel_p1_p2_response.png", dpi=150)

    # Create channels
    create_channel(p1, "config/channel_models", "Lin-Phoong_P1")
    create_channel(p2, "config/channel_models", "Lin-Phoong_P2")

    loaded_p1 = load_channel("config/channel_models/Lin-Phoong_P1.npy")
    print(loaded_p1)
    loaded_p2 = load_channel("config/channel_models/Lin-Phoong_P2.npy")
    print(loaded_p2)
