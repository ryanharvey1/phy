"""Event-triggered PSTH view for phy."""

import logging
from pathlib import Path

import numpy as np
import scipy.io as sio

from phylib.utils import Bunch
from phy.cluster.views.base import ManualClusteringView, ScalingMixin
from phy.plot.visuals import HistogramVisual, LineVisual, TextVisual
from phy.utils.color import selected_cluster_color
from phy import IPlugin, connect

logger = logging.getLogger(__name__)


def crossCorr(
    t1: np.ndarray,
    t2: np.ndarray,
    binsize: float,
    nbins: int,
) -> np.ndarray:
    """
    Performs the discrete cross-correlogram of two time series.
    The units should be in s for all arguments.
    Return the firing rate of the series t2 relative to the timings of t1.

    crossCorr functions from Guillaume Viejo of Peyrache Lab
    https://github.com/PeyracheLab/StarterPack/blob/master/python/main6_autocorr.py
    https://github.com/pynapple-org/pynapple/blob/main/pynapple/process/correlograms.py

    Parameters
    ----------
    t1 : array
        First time series.
    t2 : array
        Second time series.
    binsize : float
        Size of the bin in seconds.
    nbins : int
        Number of bins.

    Returns
    -------
    C : array
        Cross-correlogram of the two time series.

    """
    # Calculate the length of the input time series
    nt1 = len(t1)
    nt2 = len(t2)

    # Ensure that 'nbins' is an odd number
    if np.floor(nbins / 2) * 2 == nbins:
        nbins = nbins + 1

    # Calculate the half-width of the cross-correlogram window
    w = (nbins / 2) * binsize
    C = np.zeros(nbins)
    i2 = 1

    # Iterate through the first time series
    for i1 in range(nt1):
        lbound = t1[i1] - w

        # Find the index of the first element in 't2' that is within 'lbound'
        while i2 < nt2 and t2[i2] < lbound:
            i2 = i2 + 1

        # Find the index of the last element in 't2' that is within 'lbound'
        while i2 > 1 and t2[i2 - 1] > lbound:
            i2 = i2 - 1

        rbound = lbound
        idx = i2

        # Calculate the cross-correlogram values for each bin
        for j in range(nbins):
            k = 0
            rbound = rbound + binsize

            # Count the number of elements in 't2' that fall within the bin
            while idx < nt2 and t2[idx] < rbound:
                idx = idx + 1
                k = k + 1

            C[j] += k

    # Normalize the cross-correlogram by dividing by the total observation time and bin size
    C = C / (nt1 * binsize)

    return C


def compute_psth(
    spikes: list,
    events: np.ndarray,
    bin_width: float = 0.002,
    window: list = None,
):
    """Compute PSTH for spike trains around events.

    Parameters
    ----------
    spikes : list
        List of spike time arrays for each cluster.
    events : np.ndarray
        Array of event timestamps.
    bin_width : float
        Bin width in seconds.
    window : list
        Time window around events [start, end] in seconds.

    Returns
    -------
    psth_data : list
        List of PSTH arrays for each spike train.
    times : np.ndarray
        Time bins for the PSTH.
    """
    if window is None:
        window = [-0.5, 0.5]

    # Create time bins
    times = np.arange(window[0], window[1] + bin_width / 2, bin_width)
    n_bins = len(times) - 1

    psth_data = []
    for spike_times in spikes:
        psth = crossCorr(events, spike_times, bin_width, n_bins)
        psth_data.append(psth)

    return psth_data, times[:-1]  # Return bin centers


class EventPsthView(ScalingMixin, ManualClusteringView):
    """A view showing the peri-stimulus time histogram (PSTH) of spikes around events.

    This view displays PSTH plots for selected clusters, showing firing rate changes
    around specific events loaded from a .mat file.
    """

    # View configuration
    max_n_clusters = 10
    _default_position = "right"
    cluster_ids = ()

    # PSTH parameters
    bin_width = 0.005  # Bin width in seconds
    window_start = -1.0  # Window start in seconds
    window_end = 1.0  # Window end in seconds

    # UI shortcuts and snippets
    default_shortcuts = {
        "change_bin_size": "alt+wheel",
        "change_window_size": "ctrl+wheel",
    }

    default_snippets = {
        "set_bin_width": "pb",
        "set_window": "pw",
        "reload_events": "pr",
    }

    def __init__(self, get_spike_times=None, **kwargs):
        super(EventPsthView, self).__init__(**kwargs)

        # State attributes to save
        self.state_attrs += ("bin_width", "window_start", "window_end")
        self.local_state_attrs += ()

        # Set up canvas layout
        self.canvas.set_layout(layout="stacked", n_plots=1)
        self.canvas.enable_axes()

        # Function to get spike times for clusters
        self.get_spike_times = get_spike_times

        # Initialize visuals
        self.histogram_visual = HistogramVisual()
        self.canvas.add_visual(self.histogram_visual)

        self.line_visual = LineVisual()
        self.canvas.add_visual(self.line_visual)

        self.text_visual = TextVisual(color=(1.0, 1.0, 1.0, 1.0))
        self.canvas.add_visual(self.text_visual)

        # Event data
        self._events = None
        self._event_filename = None

        # Load events on initialization
        self._load_events()

    def _load_events(self):
        """Load event data from .mat file in current directory."""
        try:
            # Find file with "events" in the name
            current_dir = Path.cwd()
            event_files = [
                f for f in current_dir.glob("*.mat") if "events" in f.name.lower()
            ]

            if not event_files:
                logger.warning("No event files found in current directory")
                return

            # Use the first event file found
            filename = event_files[0]
            self._event_filename = filename.name

            data = sio.loadmat(filename, simplify_cells=True)
            event_keys = [key for key in data.keys() if not key.startswith("__")]

            if not event_keys:
                logger.warning(f"No event data found in {filename}")
                return

            # Use the last event key (following original code pattern)
            event_key = event_keys[-1]
            logger.info(f"Loading events from {filename}, key: {event_key}")

            events_data = data[event_key]
            if isinstance(events_data, dict) and "timestamps" in events_data:
                self._events = events_data["timestamps"]
                if self._events.ndim > 1:
                    self._events = self._events[:, 0]  # Take first column if 2D
            else:
                self._events = np.asarray(events_data)

            logger.info(f"Loaded {len(self._events)} events from {filename}")

        except Exception as e:
            logger.error(f"Error loading events: {e}")
            self._events = None

    def get_clusters_data(self, load_all=None):
        """Return PSTH data for selected clusters."""
        if not self.cluster_ids or self._events is None or not self.get_spike_times:
            return []

        # Get spike times for each cluster
        spike_trains = []
        for cluster_id in self.cluster_ids:
            spike_times = self.get_spike_times(cluster_id)
            if spike_times is not None and len(spike_times) > 0:
                spike_trains.append(spike_times)
            else:
                spike_trains.append(np.array([]))

        # Compute PSTH
        window = [self.window_start, self.window_end]
        psth_data, time_bins = compute_psth(
            spike_trains, self._events, self.bin_width, window
        )

        # Create bunch objects for each cluster
        result_bunchs = []
        max_rate = max(max(psth) if len(psth) > 0 else 0 for psth in psth_data)
        if max_rate == 0:
            max_rate = 1.0

        for i, (cluster_id, psth) in enumerate(zip(self.cluster_ids, psth_data)):
            bunch = Bunch()
            bunch.histogram = psth
            bunch.time_bins = time_bins
            bunch.color = selected_cluster_color(i, alpha=0.8)
            bunch.cluster_id = cluster_id
            bunch.index = i
            bunch.ylim = max_rate
            bunch.data_bounds = (time_bins[0], 0, time_bins[-1], max_rate)
            result_bunchs.append(bunch)

        return result_bunchs

    def _plot_cluster(self, bunch):
        """Plot PSTH for one cluster."""
        # Plot histogram
        self.histogram_visual.add_batch_data(
            hist=bunch.histogram,
            color=bunch.color,
            ylim=bunch.ylim,
            box_index=bunch.index,
        )

        # Add vertical line at t=0 (event time)
        zero_line_pos = np.array([[0, 0, 0, bunch.ylim]])
        self.line_visual.add_batch_data(
            pos=zero_line_pos,
            color=(0.8, 0.8, 0.8, 1.0),
            data_bounds=bunch.data_bounds,
            box_index=bunch.index,
        )

        # Add cluster info text
        info_text = f"Cluster {bunch.cluster_id}"
        if self._event_filename:
            info_text += f"\nEvents: {self._event_filename}"
        info_text += f"\nBin: {self.bin_width * 1000:.1f}ms"
        info_text += f"\nWindow: [{self.window_start:.1f}, {self.window_end:.1f}]s"

        self.text_visual.add_batch_data(
            text=[info_text],
            pos=[(-0.95, 0.9)],
            anchor=[(1, -1)],
            box_index=bunch.index,
        )

    # User interaction methods
    def on_mouse_wheel(self, e):
        """Handle mouse wheel events for changing parameters."""
        super(EventPsthView, self).on_mouse_wheel(e)

        if "Alt" in e.modifiers:
            # Change bin size
            if e.delta > 0:
                self.bin_width = min(0.1, self.bin_width * 1.2)
            else:
                self.bin_width = max(0.001, self.bin_width / 1.2)
            logger.info(f"Bin width: {self.bin_width * 1000:.1f}ms")
            self.plot()
        elif "Control" in e.modifiers:
            # Change window size
            window_size = self.window_end - self.window_start
            if e.delta > 0:
                new_size = min(10.0, window_size * 1.2)
            else:
                new_size = max(0.1, window_size / 1.2)

            # Keep window centered
            center = (self.window_start + self.window_end) / 2
            self.window_start = center - new_size / 2
            self.window_end = center + new_size / 2

            logger.info(f"Window: [{self.window_start:.1f}, {self.window_end:.1f}]s")
            self.plot()

    # Snippet methods for text commands
    def set_bin_width(self, width_ms):
        """Set bin width in milliseconds."""
        try:
            self.bin_width = float(width_ms) / 1000.0
            self.bin_width = np.clip(self.bin_width, 0.001, 0.1)
            logger.info(f"Bin width set to {self.bin_width * 1000:.1f}ms")
            self.plot()
        except ValueError:
            logger.error("Invalid bin width value")

    def set_window(self, window_str):
        """Set time window. Format: 'start,end' in seconds."""
        try:
            parts = window_str.split(",")
            if len(parts) == 2:
                start, end = float(parts[0]), float(parts[1])
                if start < end:
                    self.window_start, self.window_end = start, end
                    logger.info(f"Window set to [{start:.1f}, {end:.1f}]s")
                    self.plot()
                else:
                    logger.error("Window start must be less than end")
            else:
                logger.error("Window format: 'start,end' (e.g., '-1,1')")
        except ValueError:
            logger.error("Invalid window values")

    def reload_events(self):
        """Reload event data from file."""
        self._load_events()
        if self._events is not None:
            logger.info(f"Reloaded {len(self._events)} events")
            self.plot()
        else:
            logger.warning("Failed to reload events")


class EventPsthPlugin(IPlugin):
    """Plugin to add EventPsthView to the GUI."""

    def attach_to_controller(self, controller):
        def get_spike_times_for_cluster(cluster_id):
            """Get spike times for a single cluster."""
            try:
                # Get spike IDs for this cluster
                spike_ids = controller.supervisor.clustering.spike_ids[cluster_id]
                # Get spike times for these IDs
                spike_times = controller.model.spike_times[spike_ids]
                return spike_times
            except (KeyError, AttributeError, IndexError) as e:
                logger.warning(
                    f"Could not get spike times for cluster {cluster_id}: {e}"
                )
                return np.array([])

        def create_event_psth_view():
            """Create an EventPsthView instance."""
            return EventPsthView(get_spike_times=get_spike_times_for_cluster)

        # Store the view creation function in the controller
        controller.create_event_psth_view = create_event_psth_view

        @connect
        def on_gui_ready(sender, gui):
            # Add the view creation function to the gui's view creators
            gui.view_creator["EventPsthView"] = create_event_psth_view

            # Add the view to the GUI actions
            @gui.view_actions.add(shortcut="alt+shift+p")
            def show_event_psth():
                """Show or create the Event PSTH view."""
                gui.show_view("EventPsthView")
