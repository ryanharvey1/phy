"""
Example script showing how to create test events for the EventPsthView plugin.

This script creates a sample events.mat file that can be used to test
the EventPsthView functionality.
"""

import numpy as np
import scipy.io as sio


def create_test_events(filename="test_events.mat", duration=3600, n_events=100):
    """
    Create a test events file for the EventPsthView plugin.

    Parameters
    ----------
    filename : str
        Name of the output .mat file
    duration : float
        Total duration in seconds
    n_events : int
        Number of events to create
    """

    # Generate random event times
    event_times = np.sort(np.random.uniform(0, duration, n_events))

    # Create the events structure that the plugin expects
    events_data = {
        "events": {
            "timestamps": event_times.reshape(-1, 1)  # Column vector
        }
    }

    # Save to .mat file
    sio.savemat(filename, events_data)
    print(f"Created {filename} with {n_events} events over {duration}s")
    print(f"Event times range: {event_times[0]:.2f}s to {event_times[-1]:.2f}s")

    return filename


def create_periodic_events(filename="periodic_events.mat", duration=3600, interval=30):
    """
    Create periodic events for testing.

    Parameters
    ----------
    filename : str
        Name of the output .mat file
    duration : float
        Total duration in seconds
    interval : float
        Interval between events in seconds
    """

    # Generate periodic event times
    event_times = np.arange(interval, duration, interval)

    # Add some jitter to make it more realistic
    jitter = np.random.normal(0, interval * 0.05, len(event_times))
    event_times = event_times + jitter
    event_times = event_times[event_times > 0]  # Remove negative times
    event_times = np.sort(event_times)

    # Create the events structure
    events_data = {"events": {"timestamps": event_times.reshape(-1, 1)}}

    # Save to .mat file
    sio.savemat(filename, events_data)
    print(f"Created {filename} with {len(event_times)} periodic events")
    print(f"Mean interval: {np.mean(np.diff(event_times)):.2f}s")

    return filename


if __name__ == "__main__":
    # Create example files
    print("Creating test event files...")

    # Random events
    create_test_events("random_events.mat", duration=1800, n_events=50)

    # Periodic events
    create_periodic_events("stimulus_events.mat", duration=1800, interval=60)

    print("\nEvent files created! You can now test the EventPsthView with these files.")
    print("\nTo use:")
    print("1. Make sure one of these .mat files is in your phy working directory")
    print("2. Load the EventPsthView plugin in phy")
    print("3. Select some clusters and press Alt+Shift+P to open the PSTH view")
