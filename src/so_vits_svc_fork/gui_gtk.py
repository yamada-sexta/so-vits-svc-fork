import json
import multiprocessing
import os
import sys
from copy import copy
from logging import getLogger
from pathlib import Path
import time

import sounddevice as sd
import soundfile as sf
import torch
from pebble import ProcessFuture, ProcessPool

# Import GTK and Adwaita
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk, Adw, GLib, Gio

# Assume these are available from your project structure
# You might need to adjust imports based on your project structure
try:
    # Attempt relative import if part of a package
    from . import __version__
    from .utils import get_optimal_device
    from .inference.main import infer, realtime
except ImportError:
    # Fallback for running as a standalone script (adjust paths if necessary)
    import sys
    sys.path.append(str(Path(__file__).parent))
    try:
        import so_vits_svc_fork
        __version__ = so_vits_svc_fork.__version__
        from so_vits_svc_fork.utils import get_optimal_device
        from so_vits_svc_fork.inference.main import infer, realtime
    except ImportError:
        print("Error: Could not import voice conversion modules.")
        print("Make sure you are running from the root of the so-vits-svc-fork project")
        print("or that the necessary modules are in your Python path.")
        sys.exit(1)


# --- Reusable Backend Functions (mostly copied) ---

GUI_DEFAULT_PRESETS_PATH = Path(__file__).parent.parent / "default_gui_presets.json" # Correct path relative to repo root
GUI_PRESETS_PATH = Path("./user_gui_presets.json").absolute()

LOG = getLogger(__name__)


def play_audio_process(path: Path | str):
    """Plays audio in a separate process."""
    if isinstance(path, Path):
        path = path.as_posix()
    try:
        data, sr = sf.read(path)
        sd.play(data, sr, blocking=True) # blocking=True to finish playback in the process
    except Exception as e:
        LOG.error(f"Error playing audio {path}: {e}")


def load_presets() -> dict:
    defaults = json.loads(GUI_DEFAULT_PRESETS_PATH.read_text("utf-8"))
    users = (
        json.loads(GUI_PRESETS_PATH.read_text("utf-8"))
        if GUI_PRESETS_PATH.exists()
        else {}
    )
    # priority: users > defaults
    # order: defaults -> users
    # Note: PySimpleGUI version had priority defaults > users, this is unusual.
    # Standard practice is user settings override defaults. Adjusting here.
    return {**defaults, **users}


def add_preset(name: str, preset: dict) -> dict:
    presets = load_presets()
    presets[name] = preset
    try:
        with GUI_PRESETS_PATH.open("w") as f:
            json.dump(presets, f, indent=2)
    except Exception as e:
        LOG.error(f"Error saving preset {name}: {e}")
    return load_presets()


def delete_preset(name: str) -> dict:
    presets = load_presets()
    if name in presets:
        del presets[name]
        try:
            with GUI_PRESETS_PATH.open("w") as f:
                json.dump(presets, f, indent=2)
        except Exception as e:
            LOG.error(f"Error deleting preset {name}: {e}")
    else:
        LOG.warning(f"Cannot delete preset {name} because it does not exist.")
    return load_presets()


def get_output_path(input_path: Path) -> Path:
    # Default output path
    output_path = input_path.parent / f"{input_path.stem}.out{input_path.suffix}"

    # Increment file number in path if output file already exists
    file_num = 1
    while output_path.exists():
        output_path = (
            input_path.parent / f"{input_path.stem}.out_{file_num}{input_path.suffix}"
        )
        file_num += 1
    return output_path


# Corrected type hint and usage of Gtk.FileFilter
def get_supported_file_types() -> list[Gtk.FileFilter]:
    """Returns a list of Gtk.FileFilter objects."""
    filters = []
    sf_formats = sf.available_formats()
    # Add an "All supported audio" filter
    all_supported_glob = " ".join([f"*.{ext.lower()}" for ext in sf_formats.keys()])
    all_filter = Gtk.FileFilter.new() # Use Gtk.FileFilter
    all_filter.set_name("All supported audio")
    all_filter.add_pattern(all_supported_glob)
    filters.append(all_filter)

    # Add filters for individual formats
    for name, description in sf_formats.items():
        # sf_formats description is often empty, use name as fallback
        file_filter = Gtk.FileFilter.new() # Use Gtk.FileFilter
        filter_name = f"{name} ({description or name} files)"
        file_filter.set_name(filter_name)
        file_filter.add_pattern(f"*.{name.lower()}")
        filters.append(file_filter)

    return filters

# validate_output_file_type logic will be handled by the file dialog itself
# and potentially extra checks on the entered text


def get_devices(
    update: bool = True,
) -> tuple[list[str], list[str], list[int], list[int]]:
    if update:
        sd._terminate()
        sd._initialize()
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    for hostapi in hostapis:
        for device_idx in hostapi["devices"]:
            try:
                devices[device_idx]["hostapi_name"] = hostapi["name"]
            except IndexError:
                # Handle cases where device_idx might be out of range for some reason
                LOG.warning(f"Device index {device_idx} from hostapi {hostapi['name']} is out of range.")
                pass

    input_devices_info = [
        d for d in devices if d["max_input_channels"] > 0
    ]
    output_devices_info = [
        d for d in devices if d["max_output_channels"] > 0
    ]

    input_devices = [
        f"{d['name']} ({d.get('hostapi_name', 'Unknown HostAPI')})" # Use get for safety
        for d in input_devices_info
    ]
    output_devices = [
        f"{d['name']} ({d.get('hostapi_name', 'Unknown HostAPI')})" # Use get for safety
        for d in output_devices_info
    ]
    input_devices_indices = [d["index"] for d in input_devices_info]
    output_devices_indices = [d["index"] for d in output_devices_info]

    return input_devices, output_devices, input_devices_indices, output_devices_indices


# --- GTK4 Application Window ---

class VoiceConverterWindow(Adw.ApplicationWindow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.set_title(f"Voice Converter GUI v{__version__}")
        self.set_default_size(800, 700) # Adjusted default size

        # Use Adw.PreferencesWindow/Page/Group/Row for a modern look
        self.main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.set_content(self.main_box)

        self.header_bar = Gtk.HeaderBar()
        self.main_box.append(self.header_bar)

        self.prefs_window = Adw.PreferencesWindow()
        self.prefs_window.set_search_enabled(False) # Disable search for simplicity

        self.prefs_page = Adw.PreferencesPage()
        self.prefs_window.add(self.prefs_page)

        # Use a ScrolledWindow to make the content scrollable if it overflows
        self.scrolled_window = Gtk.ScrolledWindow()
        self.scrolled_window.set_vexpand(True)
        self.scrolled_window.set_hexpand(True)
        self.scrolled_window.set_child(self.prefs_window) # Put the prefs window content inside the scrollable area
        self.main_box.append(self.scrolled_window)


        # Store widgets we need to access later
        self.widgets = {}
        self.process_pool: ProcessPool | None = None
        self.realtime_future: ProcessFuture | None = None
        self.infer_futures: set[ProcessFuture] = set()
        self.audio_play_future: ProcessFuture | None = None # Track play audio future

        # Store device indices mapping string name to sd index
        self._input_device_indices: list[int] = []
        self._output_device_indices: list[int] = []

        self._build_ui()
        self._load_initial_data()
        self._connect_signals()
        self._setup_process_pool()
        self._setup_timeout_check()


    def _build_ui(self):
        # --- Paths Group ---
        paths_group = Adw.PreferencesGroup.new()
        paths_group.set_title("Paths")
        self.prefs_page.add(paths_group)

        # Model Path
        self.widgets["model_path"] = Gtk.Entry()
        model_candidates = list(sorted(Path("./logs/44k/").glob("G_*.pth")))
        if model_candidates:
            self.widgets["model_path"].set_text(model_candidates[-1].absolute().as_posix())
        model_path_row = Adw.ActionRow.new()
        model_path_row.set_title("Model path")
        model_path_row.add_suffix(self.widgets["model_path"])
        model_browse_button = Gtk.Button(label="Browse")
        model_browse_button.connect("clicked", self._on_browse_clicked, "model_path", "open",
                                    [Gtk.FileFilter.new().add_pattern("G_*.pth"), Gtk.FileFilter.new().add_pattern("G_*.pt")]) # Use Gtk.FileFilter
        model_path_row.add_suffix(model_browse_button)
        paths_group.add(model_path_row)


        # Config Path
        self.widgets["config_path"] = Gtk.Entry()
        if Path("./configs/44k/config.json").exists():
            self.widgets["config_path"].set_text(Path("./configs/44k/config.json").absolute().as_posix())
        config_path_row = Adw.ActionRow.new()
        config_path_row.set_title("Config path")
        config_path_row.add_suffix(self.widgets["config_path"])
        config_browse_button = Gtk.Button(label="Browse")
        config_browse_button.connect("clicked", self._on_browse_clicked, "config_path", "open",
                                     [Gtk.FileFilter.new().add_pattern("*.json")]) # Use Gtk.FileFilter
        config_path_row.add_suffix(config_browse_button)
        paths_group.add(config_path_row)

        # Cluster Model Path
        self.widgets["cluster_model_path"] = Gtk.Entry()
        if Path("./logs/44k/kmeans.pt").exists():
             self.widgets["cluster_model_path"].set_text(Path("./logs/44k/kmeans.pt").absolute().as_posix())
        cluster_path_row = Adw.ActionRow.new()
        cluster_path_row.set_title("Cluster model path (Optional)")
        cluster_path_row.add_suffix(self.widgets["cluster_model_path"])
        cluster_browse_button = Gtk.Button(label="Browse")
        cluster_browse_button.connect("clicked", self._on_browse_clicked, "cluster_model_path", "open",
                                      [Gtk.FileFilter.new().add_pattern("*.pt"), Gtk.FileFilter.new().add_pattern("*.pth"), Gtk.FileFilter.new().add_pattern("*.pkl")]) # Use Gtk.FileFilter
        cluster_path_row.add_suffix(cluster_browse_button)
        paths_group.add(cluster_path_row)


        # --- Common Group ---
        common_group = Adw.PreferencesGroup.new()
        common_group.set_title("Common Settings")
        self.prefs_page.add(common_group)

        # Speaker
        self.widgets["speaker"] = Gtk.ComboBoxText()
        speaker_row = Adw.ActionRow.new()
        speaker_row.set_title("Speaker")
        speaker_row.add_suffix(self.widgets["speaker"])
        common_group.add(speaker_row)

        # Silence Threshold
        self.widgets["silence_threshold_adj"] = Gtk.Adjustment.new(0.0, -60.0, 0.0, 0.1, 1.0, 0.0)
        self.widgets["silence_threshold"] = Gtk.Scale.new(Gtk.Orientation.HORIZONTAL, self.widgets["silence_threshold_adj"])
        self.widgets["silence_threshold"].set_digits(1)
        self.widgets["silence_threshold"].set_hexpand(True)
        silence_row = Adw.ActionRow.new()
        silence_row.set_title("Silence threshold (dB)")
        silence_row.add_suffix(self.widgets["silence_threshold"])
        common_group.add(silence_row)

        # Transpose
        self.widgets["transpose_adj"] = Gtk.Adjustment.new(0, -36, 36, 1, 12, 0)
        self.widgets["transpose"] = Gtk.Scale.new(Gtk.Orientation.HORIZONTAL, self.widgets["transpose_adj"])
        self.widgets["transpose"].set_digits(0)
        # self.widgets["transpose"].set_marks([-36, -24, -12, 0, 12, 24, 36], Gtk.PositionType.BOTTOM)
        self.widgets["transpose"].set_hexpand(True)
        transpose_row = Adw.ActionRow.new()
        transpose_row.set_title("Pitch (12 = 1 octave)")
        transpose_row.set_subtitle("ADJUST THIS based on your voice when Auto predict F0 is turned off.")
        transpose_row.add_suffix(self.widgets["transpose"])
        common_group.add(transpose_row)

        # Auto Predict F0
        self.widgets["auto_predict_f0"] = Gtk.CheckButton.new_with_label(
            "Auto predict F0 (Pitch may become unstable when turned on in real-time inference.)"
        )
        auto_f0_row = Adw.ActionRow.new()
        auto_f0_row.add_suffix(self.widgets["auto_predict_f0"])
        common_group.add(auto_f0_row)


        # F0 Prediction Method
        self.widgets["f0_method"] = Gtk.ComboBoxText()
        for method in ["crepe", "crepe-tiny", "parselmouth", "dio", "harvest"]:
            self.widgets["f0_method"].append_text(method)
        self.widgets["f0_method"].set_active(0) # Default to crepe
        f0_method_row = Adw.ActionRow.new()
        f0_method_row.set_title("F0 prediction method")
        f0_method_row.add_suffix(self.widgets["f0_method"])
        common_group.add(f0_method_row)

        # Cluster Infer Ratio
        self.widgets["cluster_infer_ratio_adj"] = Gtk.Adjustment.new(0.0, 0.0, 1.0, 0.01, 0.1, 0.0)
        self.widgets["cluster_infer_ratio"] = Gtk.Scale.new(Gtk.Orientation.HORIZONTAL, self.widgets["cluster_infer_ratio_adj"])
        self.widgets["cluster_infer_ratio"].set_digits(2)
        self.widgets["cluster_infer_ratio"].set_hexpand(True)
        cluster_ratio_row = Adw.ActionRow.new()
        cluster_ratio_row.set_title("Cluster infer ratio")
        cluster_ratio_row.add_suffix(self.widgets["cluster_infer_ratio"])
        common_group.add(cluster_ratio_row)

        # Noise Scale
        self.widgets["noise_scale_adj"] = Gtk.Adjustment.new(0.4, 0.0, 1.0, 0.01, 0.1, 0.0) # Default 0.4? PySimpleGUI didn't specify
        self.widgets["noise_scale"] = Gtk.Scale.new(Gtk.Orientation.HORIZONTAL, self.widgets["noise_scale_adj"])
        self.widgets["noise_scale"].set_digits(2)
        self.widgets["noise_scale"].set_hexpand(True)
        noise_scale_row = Adw.ActionRow.new()
        noise_scale_row.set_title("Noise scale")
        noise_scale_row.add_suffix(self.widgets["noise_scale"])
        common_group.add(noise_scale_row)

        # Pad Seconds
        self.widgets["pad_seconds_adj"] = Gtk.Adjustment.new(0.0, 0.0, 1.0, 0.01, 0.1, 0.0) # Default 0.0?
        self.widgets["pad_seconds"] = Gtk.Scale.new(Gtk.Orientation.HORIZONTAL, self.widgets["pad_seconds_adj"])
        self.widgets["pad_seconds"].set_digits(2)
        self.widgets["pad_seconds"].set_hexpand(True)
        pad_seconds_row = Adw.ActionRow.new()
        pad_seconds_row.set_title("Pad seconds")
        pad_seconds_row.add_suffix(self.widgets["pad_seconds"])
        common_group.add(pad_seconds_row)

        # Chunk Seconds
        self.widgets["chunk_seconds_adj"] = Gtk.Adjustment.new(0.0, 0.0, 3.0, 0.01, 0.1, 0.0) # Default 0.0?
        self.widgets["chunk_seconds"] = Gtk.Scale.new(Gtk.Orientation.HORIZONTAL, self.widgets["chunk_seconds_adj"])
        self.widgets["chunk_seconds"].set_digits(2)
        self.widgets["chunk_seconds"].set_hexpand(True)
        chunk_seconds_row = Adw.ActionRow.new()
        chunk_seconds_row.set_title("Chunk seconds")
        chunk_seconds_row.add_suffix(self.widgets["chunk_seconds"])
        common_group.add(chunk_seconds_row)

        # Max Chunk Seconds
        self.widgets["max_chunk_seconds_adj"] = Gtk.Adjustment.new(0.0, 0.0, 240.0, 1.0, 10.0, 0.0) # Default 0.0?
        self.widgets["max_chunk_seconds"] = Gtk.Scale.new(Gtk.Orientation.HORIZONTAL, self.widgets["max_chunk_seconds_adj"])
        self.widgets["max_chunk_seconds"].set_digits(0)
        self.widgets["max_chunk_seconds"].set_hexpand(True)
        max_chunk_seconds_row = Adw.ActionRow.new()
        max_chunk_seconds_row.set_title("Max chunk seconds")
        max_chunk_seconds_row.set_subtitle("(set lower if Out Of Memory, 0 to disable)")
        max_chunk_seconds_row.add_suffix(self.widgets["max_chunk_seconds"])
        common_group.add(max_chunk_seconds_row)

        # Absolute Threshold
        self.widgets["absolute_thresh"] = Gtk.CheckButton.new_with_label(
            "Absolute threshold (ignored in realtime inference)"
        )
        abs_thresh_row = Adw.ActionRow.new()
        abs_thresh_row.add_suffix(self.widgets["absolute_thresh"])
        common_group.add(abs_thresh_row)


        # --- File Group ---
        file_group = Adw.PreferencesGroup.new()
        file_group.set_title("File Inference")
        self.prefs_page.add(file_group)

        # Input Audio Path
        self.widgets["input_path"] = Gtk.Entry()
        input_path_row = Adw.ActionRow.new()
        input_path_row.set_title("Input audio path")
        input_path_row.add_suffix(self.widgets["input_path"])
        input_browse_button = Gtk.Button(label="Browse File")
        input_browse_button.connect("clicked", self._on_browse_clicked, "input_path", "open",
                                    get_supported_file_types())
        input_path_row.add_suffix(input_browse_button)
        input_folder_button = Gtk.Button(label="Browse Folder")
        input_folder_button.connect("clicked", self._on_browse_clicked, "input_path", "folder", None) # No filters for folder
        input_path_row.add_suffix(input_folder_button)
        file_group.add(input_path_row)

        # Play Input Button
        play_input_button = Gtk.Button(label="Play Input Audio")
        play_input_button.connect("clicked", self._on_play_input_clicked)
        play_row = Adw.ActionRow.new()
        play_row.add_suffix(play_input_button)
        file_group.add(play_row)

        # Output Audio Path
        self.widgets["output_path"] = Gtk.Entry()
        output_path_row = Adw.ActionRow.new()
        output_path_row.set_title("Output audio path")
        output_path_row.add_suffix(self.widgets["output_path"])
        output_save_button = Gtk.Button(label="Save As")
        output_save_button.connect("clicked", self._on_browse_clicked, "output_path", "save",
                                   get_supported_file_types())
        output_path_row.add_suffix(output_save_button)
        file_group.add(output_path_row)

        # Auto Play
        self.widgets["auto_play"] = Gtk.CheckButton.new_with_label("Auto play output after inference")
        self.widgets["auto_play"].set_active(True)
        auto_play_row = Adw.ActionRow.new()
        auto_play_row.add_suffix(self.widgets["auto_play"])
        file_group.add(auto_play_row)

        # Infer Button
        self.widgets["infer"] = Gtk.Button(label="Start File Inference")
        self.widgets["infer"].set_halign(Gtk.Align.CENTER)
        self.widgets["infer"].set_vexpand(False) # Don't expand vertically
        file_group.add(self.widgets["infer"])


        # --- Realtime Group ---
        realtime_group = Adw.PreferencesGroup.new()
        realtime_group.set_title("Realtime Voice Changer")
        self.prefs_page.add(realtime_group)

        # Crossfade Seconds
        self.widgets["crossfade_seconds_adj"] = Gtk.Adjustment.new(0.0, 0.0, 0.6, 0.001, 0.01, 0.0)
        self.widgets["crossfade_seconds"] = Gtk.Scale.new(Gtk.Orientation.HORIZONTAL, self.widgets["crossfade_seconds_adj"])
        self.widgets["crossfade_seconds"].set_digits(3)
        self.widgets["crossfade_seconds"].set_hexpand(True)
        crossfade_row = Adw.ActionRow.new()
        crossfade_row.set_title("Crossfade seconds")
        crossfade_row.add_suffix(self.widgets["crossfade_seconds"])
        realtime_group.add(crossfade_row)

        # Block Seconds
        self.widgets["block_seconds_adj"] = Gtk.Adjustment.new(0.0, 0.0, 3.0, 0.001, 0.01, 0.0)
        self.widgets["block_seconds"] = Gtk.Scale.new(Gtk.Orientation.HORIZONTAL, self.widgets["block_seconds_adj"])
        self.widgets["block_seconds"].set_digits(3)
        self.widgets["block_seconds"].set_hexpand(True)
        block_seconds_row = Adw.ActionRow.new()
        block_seconds_row.set_title("Block seconds")
        block_seconds_row.set_subtitle("Big -> more robust, slower, (the same) latency")
        block_seconds_row.add_suffix(self.widgets["block_seconds"])
        realtime_group.add(block_seconds_row)

        # Additional Infer Before Seconds
        self.widgets["additional_infer_before_seconds_adj"] = Gtk.Adjustment.new(0.0, 0.0, 2.0, 0.001, 0.01, 0.0)
        self.widgets["additional_infer_before_seconds"] = Gtk.Scale.new(Gtk.Orientation.HORIZONTAL, self.widgets["additional_infer_before_seconds_adj"])
        self.widgets["additional_infer_before_seconds"].set_digits(3)
        self.widgets["additional_infer_before_seconds"].set_hexpand(True)
        infer_before_row = Adw.ActionRow.new()
        infer_before_row.set_title("Additional Infer seconds (before)")
        infer_before_row.set_subtitle("Big -> more robust, slower, additional latency")
        infer_before_row.add_suffix(self.widgets["additional_infer_before_seconds"])
        realtime_group.add(infer_before_row)

        # Additional Infer After Seconds
        self.widgets["additional_infer_after_seconds_adj"] = Gtk.Adjustment.new(0.0, 0.0, 2.0, 0.001, 0.01, 0.0)
        self.widgets["additional_infer_after_seconds"] = Gtk.Scale.new(Gtk.Orientation.HORIZONTAL, self.widgets["additional_infer_after_seconds_adj"])
        self.widgets["additional_infer_after_seconds"].set_digits(3)
        self.widgets["additional_infer_after_seconds"].set_hexpand(True)
        infer_after_row = Adw.ActionRow.new()
        infer_after_row.set_title("Additional Infer seconds (after)")
        infer_after_row.set_subtitle("Big -> more robust, slower, additional latency")
        infer_after_row.add_suffix(self.widgets["additional_infer_after_seconds"])
        realtime_group.add(infer_after_row)

        # Realtime Algorithm
        self.widgets["realtime_algorithm"] = Gtk.ComboBoxText()
        self.widgets["realtime_algorithm"].append_text("1 (Divide constantly)")
        self.widgets["realtime_algorithm"].append_text("2 (Divide by speech)")
        self.widgets["realtime_algorithm"].set_active(0) # Default
        algorithm_row = Adw.ActionRow.new()
        algorithm_row.set_title("Realtime algorithm")
        algorithm_row.add_suffix(self.widgets["realtime_algorithm"])
        realtime_group.add(algorithm_row)


        # Device Selection using GtkComboBoxText for simplicity
        self.widgets["input_device"] = Gtk.ComboBoxText()
        input_device_row = Adw.ActionRow.new()
        input_device_row.set_title("Input device")
        input_device_row.add_suffix(self.widgets["input_device"])
        realtime_group.add(input_device_row)

        self.widgets["output_device"] = Gtk.ComboBoxText()
        output_device_row = Adw.ActionRow.new()
        output_device_row.set_title("Output device")
        output_device_row.add_suffix(self.widgets["output_device"])
        realtime_group.add(output_device_row)

        # Passthrough and Refresh Devices
        actions_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        self.widgets["passthrough_original"] = Gtk.CheckButton.new_with_label("Passthrough original audio (for latency check)")
        self.widgets["passthrough_original"].set_active(False)
        actions_box.append(self.widgets["passthrough_original"])
        actions_box.set_halign(Gtk.Align.END) # Push to end

        refresh_devices_button = Gtk.Button(label="Refresh devices")
        refresh_devices_button.connect("clicked", self._on_refresh_devices_clicked)
        actions_box.append(refresh_devices_button)

        actions_row = Adw.ActionRow.new()
        actions_row.add_suffix(actions_box)
        realtime_group.add(actions_row)


        # Realtime Notes Frame (using GtkFrame inside AdwPreferencesGroup)
        notes_frame = Gtk.Frame(label="Notes")
        notes_label = Gtk.Label()
        notes_label.set_markup(
            "<small>"
            "In Realtime Inference:\n"
            "    - Setting F0 prediction method to 'crepe` may cause performance degradation.\n"
            "    - Auto Predict F0 must be turned off.\n"
            "If the audio sounds mumbly and choppy:\n"
            "    Case: The inference has not been made in time (Increase Block seconds)\n"
            "    Case: Mic input is low (Decrease Silence threshold)\n"
            "</small>"
        )
        notes_label.set_wrap(True)
        notes_label.set_justify(Gtk.Justification.LEFT)
        notes_label.set_xalign(0) # Align text to the left
        notes_frame.set_child(Gtk.ScrolledWindow(child=notes_label, vexpand=True, hexpand=True))
        realtime_group.add(notes_frame)

        # Start/Stop VC Buttons
        vc_buttons_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        self.widgets["start_vc"] = Gtk.Button(label="(Re)Start Voice Changer")
        self.widgets["stop_vc"] = Gtk.Button(label="Stop Voice Changer")
        self.widgets["stop_vc"].set_sensitive(False) # Initially disabled

        vc_buttons_box.append(self.widgets["start_vc"])
        vc_buttons_box.append(self.widgets["stop_vc"])
        vc_buttons_box.set_halign(Gtk.Align.CENTER)

        vc_buttons_row = Adw.ActionRow.new()
        vc_buttons_row.add_suffix(vc_buttons_box)
        realtime_group.add(vc_buttons_row)

        # --- Presets Group ---
        presets_group = Adw.PreferencesGroup.new()
        presets_group.set_title("Presets")
        self.prefs_page.add(presets_group)

        # Presets ComboBox
        self.widgets["presets"] = Gtk.ComboBoxText()
        presets_list = list(load_presets().keys())
        for preset_name in presets_list:
             self.widgets["presets"].append_text(preset_name)
        if presets_list:
            self.widgets["presets"].set_active(0) # Default to first preset
        presets_row = Adw.ActionRow.new()
        presets_row.set_title("Presets")
        presets_row.add_suffix(self.widgets["presets"])

        delete_preset_button = Gtk.Button(label="Delete Preset")
        delete_preset_button.connect("clicked", self._on_delete_preset_clicked)
        presets_row.add_suffix(delete_preset_button)
        presets_group.add(presets_row)


        # Add Preset
        self.widgets["preset_name"] = Gtk.Entry()
        self.widgets["preset_name"].set_placeholder_text("Enter preset name")
        add_preset_row = Adw.ActionRow.new()
        add_preset_row.set_title("Preset name")
        add_preset_row.add_suffix(self.widgets["preset_name"])

        add_preset_button = Gtk.Button(label="Add current settings as a preset")
        add_preset_button.connect("clicked", self._on_add_preset_clicked)
        add_preset_row.add_suffix(add_preset_button)
        presets_group.add(add_preset_row)

        # --- GPU Checkbox ---
        # This checkbox is outside of the standard Adw layout for simplicity,
        # maybe put it in the header bar or a separate general settings group if needed.
        # Placing it below the scrollable area for now.
        self.widgets["use_gpu"] = Gtk.CheckButton.new_with_label("Use GPU")
        optimal_device = get_optimal_device()
        gpu_available = optimal_device != torch.device("cpu")
        self.widgets["use_gpu"].set_active(gpu_available)
        self.widgets["use_gpu"].set_sensitive(gpu_available) # Disable if no GPU

        if not gpu_available:
             gpu_label = self.widgets["use_gpu"].get_label_widget()
             if gpu_label and isinstance(gpu_label, Gtk.Label):
                 gpu_label.set_tooltip_text("GPU not available; if your device has GPU, make sure you installed PyTorch with CUDA support")


        gpu_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        gpu_box.append(self.widgets["use_gpu"])
        gpu_box.set_margin_start(10)
        gpu_box.set_margin_end(10)
        gpu_box.set_margin_bottom(10)

        self.main_box.append(gpu_box)

        # Store sliders for easier access to their adjustments
        self.sliders = {
            "silence_threshold": self.widgets["silence_threshold_adj"],
            "transpose": self.widgets["transpose_adj"],
            "cluster_infer_ratio": self.widgets["cluster_infer_ratio_adj"],
            "noise_scale": self.widgets["noise_scale_adj"],
            "pad_seconds": self.widgets["pad_seconds_adj"],
            "chunk_seconds": self.widgets["chunk_seconds_adj"],
            "max_chunk_seconds": self.widgets["max_chunk_seconds_adj"],
            "crossfade_seconds": self.widgets["crossfade_seconds_adj"],
            "additional_infer_before_seconds": self.widgets["additional_infer_before_seconds_adj"],
            "additional_infer_after_seconds": self.widgets["additional_infer_after_seconds_adj"],
        }


    def _load_initial_data(self):
        """Populate initial data into widgets."""
        self._update_speaker_combo()
        self._update_devices_combo()
        self._apply_preset(self.widgets["presets"].get_active_text()) # Apply the default preset


    def _connect_signals(self):
        """Connect signals to handler methods."""
        self.widgets["model_path"].connect("changed", self._on_path_entry_changed)
        self.widgets["config_path"].connect("changed", self._on_path_entry_changed)
        self.widgets["cluster_model_path"].connect("changed", self._on_path_entry_changed)
        self.widgets["input_path"].connect("changed", self._on_input_path_changed)

        self.widgets["auto_predict_f0"].connect("notify::active", self._on_auto_predict_f0_toggled)

        self.widgets["infer"].connect("clicked", self._on_infer_clicked)
        self.widgets["start_vc"].connect("clicked", self._on_start_vc_clicked)
        self.widgets["stop_vc"].connect("clicked", self._on_stop_vc_clicked)

        self.widgets["presets"].connect("changed", self._on_presets_changed)


    def _setup_process_pool(self):
        """Set up the process pool for background tasks."""
        # Using spawn context is important for compatibility, especially on Windows
        self.process_pool = ProcessPool(
            max_workers=min(2, multiprocessing.cpu_count()),
            context=multiprocessing.get_context("spawn"),
        )
        LOG.info("Process pool created.")

    def _setup_timeout_check(self):
        """Set up a periodic check for process pool futures."""
        # Check every 200 milliseconds
        GLib.timeout_add(200, self._check_futures)

    def _check_futures(self):
        """Checks on the status of process pool futures."""
        # Check realtime future
        if self.realtime_future and self.realtime_future.done():
            try:
                # Retrieve result to catch exceptions
                self.realtime_future.result()
                LOG.info("Realtime voice changer stopped.")
            except Exception as e:
                LOG.error("Error in realtime voice changer:")
                LOG.exception(e)
                GLib.idle_add(self._show_error_dialog, "Realtime Error", f"An error occurred in the realtime voice changer:\n\n{e}")
            finally:
                self.realtime_future = None
                GLib.idle_add(self._update_realtime_buttons) # Update buttons in main thread

        # Check inference futures
        completed_infer_futures = {f for f in self.infer_futures if f.done()}
        for future in completed_infer_futures:
            try:
                # Retrieve result to catch exceptions
                result = future.result()
                LOG.info(f"Inference task completed.")
                # The after_inference logic is now inside the callback added when scheduling
            except Exception as e:
                LOG.error("Error in inference task:")
                LOG.exception(e)
                GLib.idle_add(self._show_error_dialog, "Inference Error", f"An error occurred during inference:\n\n{e}")
            finally:
                self.infer_futures.discard(future) # Remove the future regardless of success/failure

        # Check audio play future
        if self.audio_play_future and self.audio_play_future.done():
             try:
                 self.audio_play_future.result()
             except Exception as e:
                 LOG.error(f"Error during audio playback: {e}")
                 GLib.idle_add(self._show_error_dialog, "Playback Error", f"An error occurred during audio playback:\n\n{e}")
             finally:
                 self.audio_play_future = None


        # Continue the timeout check
        return GLib.SOURCE_CONTINUE

    def _on_browse_clicked(self, button: Gtk.Button, entry_key: str, dialog_type: str, filters: list[Gtk.FileFilter] | None): # Corrected filter type hint
        """Handler for Browse buttons."""
        entry = self.widgets[entry_key]

        if dialog_type == "open":
            dialog = Gtk.FileDialog.new()
            dialog.set_title("Select File")
            if filters:
                 filters_list = Gio.ListStore.new(Gtk.FileFilter) # Use Gtk.FileFilter
                 for f in filters:
                     filters_list.append(f)
                 dialog.set_filters(filters_list)

            # Suggest initial folder based on current entry text or a default
            current_path = Path(entry.get_text())
            if current_path.exists() and current_path.is_dir():
                 initial_folder = Gio.File.new_for_path(str(current_path))
                 dialog.set_initial_folder(initial_folder)
            elif current_path.parent.exists() and current_path.parent.is_dir():
                 initial_folder = Gio.File.new_for_path(str(current_path.parent))
                 dialog.set_initial_folder(initial_folder)


            dialog.open(self, None, self._on_file_opened, entry)

        elif dialog_type == "folder":
            dialog = Gtk.FileDialog.new()
            dialog.set_title("Select Folder")
            dialog.set_modal(True) # Folder dialog is often modal

            # Suggest initial folder
            current_path = Path(entry.get_text())
            if current_path.exists() and current_path.is_dir():
                 initial_folder = Gio.File.new_for_path(str(current_path))
                 dialog.set_initial_folder(initial_folder)
            elif current_path.parent.exists() and current_path.parent.is_dir():
                 initial_folder = Gio.File.new_for_path(str(current_path.parent))
                 dialog.set_initial_folder(initial_folder)
            else:
                 # Default to current working directory if path is invalid
                 dialog.set_initial_folder(Gio.File.new_for_path(os.getcwd()))


            dialog.select_folder(self, None, self._on_folder_selected, entry)

        elif dialog_type == "save":
             dialog = Gtk.FileDialog.new()
             dialog.set_title("Save File As")
             if filters:
                  filters_list = Gio.ListStore.new(Gtk.FileFilter) # Use Gtk.FileFilter
                  for f in filters:
                      filters_list.append(f)
                  dialog.set_filters(filters_list)

             # Suggest initial name and folder
             current_path = Path(entry.get_text())
             if current_path.parent.exists() and current_path.parent.is_dir():
                  initial_folder = Gio.File.new_for_path(str(current_path.parent))
                  dialog.set_initial_folder(initial_folder)
             if current_path.name:
                  dialog.set_initial_name(current_path.name)
             else:
                  # Suggest a default name if entry is empty
                  input_path_str = self.widgets["input_path"].get_text()
                  if input_path_str:
                       input_path = Path(input_path_str)
                       suggested_output_path = get_output_path(input_path)
                       dialog.set_initial_name(suggested_output_path.name)
                       if suggested_output_path.parent.exists():
                           dialog.set_initial_folder(Gio.File.new_for_path(str(suggested_output_path.parent)))
                  else:
                       dialog.set_initial_name("output.wav") # Generic fallback


             dialog.save(self, None, self._on_file_saved, entry)


    def _on_file_opened(self, dialog: Gtk.FileDialog, result: Gio.AsyncResult, entry: Gtk.Entry):
        """Callback for Gtk.FileDialog.open."""
        try:
            file = dialog.open_finish(result)
            if file:
                entry.set_text(file.get_path())
        except Exception as e:
            LOG.error(f"Error opening file: {e}")
            self._show_error_dialog("Error", f"Failed to open file:\n{e}")

    def _on_folder_selected(self, dialog: Gtk.FileDialog, result: Gio.AsyncResult, entry: Gtk.Entry):
         """Callback for Gtk.FileDialog.select_folder."""
         try:
             folder = dialog.select_folder_finish(result)
             if folder:
                 entry.set_text(folder.get_path())
         except Exception as e:
             LOG.error(f"Error selecting folder: {e}")
             self._show_error_dialog("Error", f"Failed to select folder:\n{e}")


    def _on_file_saved(self, dialog: Gtk.FileDialog, result: Gio.AsyncResult, entry: Gtk.Entry):
        """Callback for Gtk.FileDialog.save."""
        try:
            file = dialog.save_finish(result)
            if file:
                # Validate extension before setting the text
                output_path = Path(file.get_path())
                # Basic check, GtkFileDialog filters help, but manual edit is possible
                supported_suffixes = [f".{ext.lower()}" for ext in sf.available_formats().keys()]
                if not output_path.suffix or output_path.suffix.lower() not in supported_suffixes:
                     LOG.warning(f"Saved file path '{output_path}' has an unsupported or missing extension.")
                     # Decide how to handle this - set the path and warn, or refuse?
                     # Setting and warning allows the user to manually fix.
                     self._show_info_dialog("Warning", f"The selected file path '{output_path.name}' has an unsupported or missing file extension. Please ensure you use a supported extension (e.g., .wav, .mp3).")
                     entry.set_text(file.get_path()) # Set it anyway, user can edit
                else:
                     entry.set_text(file.get_path())

        except Exception as e:
            LOG.error(f"Error saving file: {e}")
            self._show_error_dialog("Error", f"Failed to save file:\n{e}")


    def _on_path_entry_changed(self, entry: Gtk.Entry):
        """Handler for changes in path entry fields."""
        # No need to update browse button initial folders automatically with GtkFileDialog
        # The dialog itself can often remember the last location or you set an initial one.
        # We do, however, update the speaker list if config path changes.
        if entry == self.widgets["config_path"]:
             self._update_speaker_combo()

    def _on_input_path_changed(self, entry: Gtk.Entry):
        """Handler for changes in the input path entry."""
        input_path_str = entry.get_text()
        if input_path_str:
            input_path = Path(input_path_str)
            # Update output path suggestion only if the output path entry is empty
            if not self.widgets["output_path"].get_text():
                 # Check if input path is a directory or file first before generating output path
                 if input_path.exists() and (input_path.is_file() or input_path.is_dir()):
                     # If it's a directory, maybe suggest outputting to a default file name in that dir?
                     # The original code's get_output_path expects a file input.
                     # We'll generate a default based on a dummy name for dir inputs,
                     # or the actual file name for file inputs.
                     if input_path.is_file():
                         suggested_output_path = get_output_path(input_path)
                         self.widgets["output_path"].set_text(str(suggested_output_path))
                     elif input_path.is_dir():
                         # For directory input, just suggest a default file in that directory
                         default_output_file = input_path / "output.wav" # Or get_output_path(input_path / "dummy_input.wav") ?
                         # Let's stick to the original logic where output path is a single file.
                         # If the user browses a folder for input, the infer function must handle it.
                         # The output path entry implies a single output file or directory base.
                         # Let's not auto-fill the output path if input is a directory for now,
                         # as the single output path entry doesn't make sense for batch.
                         # If batch output is required, the UI/logic needs a separate output *directory* field.
                         # The original PySimpleGUI `get_output_path` function implies a single file output.
                         # Let's only auto-fill output if input is a file.
                         pass # Do nothing if input is a directory

    def _on_auto_predict_f0_toggled(self, checkbox: Gtk.CheckButton, pspec):
        """Handler for the Auto Predict F0 checkbox."""
        active = checkbox.get_active()
        self.widgets["transpose"].set_sensitive(not active)
        # GTK widgets don't have a direct 'visible' property tied to layout the same way
        # PySimpleGUI does for sliders in this specific context.
        # Making it insensitive is the standard GTK way to disable interaction.
        # If you *must* hide it, you'd remove/add it from its parent container,
        # which is more complex with AdwActionRow. Sensensitivity is simpler.


    def _on_play_input_clicked(self, button: Gtk.Button):
        """Handler for the Play Input Audio button."""
        input_path_str = self.widgets["input_path"].get_text()
        if not input_path_str:
            LOG.warning("Input path is empty. Cannot play.")
            self._show_info_dialog("Warning", "Input path is empty. Cannot play.")
            return

        input_path = Path(input_path_str)
        if not input_path.exists() or not input_path.is_file():
            LOG.warning(f"Input path {input_path} does not exist or is not a file.")
            self._show_info_dialog("Warning", f"Input path '{input_path_str}' does not exist or is not a file. Cannot play.")
            return

        if self.audio_play_future and not self.audio_play_future.done():
             LOG.info("Audio playback already in progress.")
             self._show_info_dialog("Info", "Audio playback already in progress.")
             return

        LOG.info(f"Playing audio from {input_path}")
        # Schedule play audio in a separate process
        self.audio_play_future = self.process_pool.schedule(play_audio_process, args=(input_path,))
        # No need for a done callback unless we want to update UI after playback finishes


    def _on_infer_clicked(self, button: Gtk.Button):
        """Handler for the Infer button."""
        input_path_str = self.widgets["input_path"].get_text()
        output_path_str = self.widgets["output_path"].get_text()

        if not input_path_str:
            LOG.warning("Input path is empty.")
            self._show_info_dialog("Warning", "Input path is empty.")
            return

        input_path = Path(input_path_str)
        if not input_path.exists() or not (input_path.is_file() or input_path.is_dir()):
             LOG.warning(f"Input path {input_path} does not exist or is not a file/folder.")
             self._show_info_dialog("Warning", f"Input path '{input_path_str}' does not exist or is not a file/folder.")
             return
        if input_path.is_dir() and not any(f.is_file() for f in input_path.iterdir()): # Check for at least one file in directory
             LOG.warning(f"Input folder {input_path} is empty or contains no files.")
             self._show_info_dialog("Warning", f"Input folder '{input_path_str}' is empty or contains no files.")
             return


        if not output_path_str:
            LOG.warning("Output path is empty.")
            self._show_info_dialog("Warning", "Output path is empty.")
            return

        output_path = Path(output_path_str)
        # Basic validation (more robust validation is harder without knowing the library's specifics)
        # We rely somewhat on the file dialog filters, but a manual check is good.
        supported_suffixes = [f".{ext.lower()}" for ext in sf.available_formats().keys()]
        if output_path.suffix.lower() not in supported_suffixes:
             LOG.warning(f"Output path has unsupported extension: {output_path.suffix}. Supported: {', '.join(supported_suffixes)}")
             self._show_error_dialog(
                 "Unsupported Output Type",
                 f"The output file extension '{output_path.suffix}' is not supported.\n"
                 f"Please use one of the following:\n"
                 f"{', '.join(supported_suffixes)}"
             )
             return

        # Collect current settings
        settings = self._get_current_settings()

        LOG.info("Starting inference...")
        self.widgets["infer"].set_sensitive(False) # Disable button

        try:
            infer_future = self.process_pool.schedule(
                infer,
                kwargs=dict(
                    # paths
                    model_path=Path(settings["model_path"]),
                    output_path=Path(settings["output_path"]),
                    input_path=Path(settings["input_path"]),
                    config_path=Path(settings["config_path"]),
                    recursive=True, # Assuming recursive for folder input
                    # svc config
                    speaker=settings["speaker"],
                    cluster_model_path=(
                        Path(settings["cluster_model_path"])
                        if settings["cluster_model_path"]
                        else None
                    ),
                    transpose=settings["transpose"],
                    auto_predict_f0=settings["auto_predict_f0"],
                    cluster_infer_ratio=settings["cluster_infer_ratio"],
                    noise_scale=settings["noise_scale"],
                    f0_method=settings["f0_method"],
                    # slice config
                    db_thresh=settings["silence_threshold"],
                    pad_seconds=settings["pad_seconds"],
                    chunk_seconds=settings["chunk_seconds"],
                    absolute_thresh=settings["absolute_thresh"],
                    max_chunk_seconds=settings["max_chunk_seconds"],
                    device=(
                        "cpu" if not settings["use_gpu"] else get_optimal_device()
                    ),
                ),
            )

            # Add a done callback that runs in a Pebble worker thread
            infer_future.add_done_callback(
                lambda _future: GLib.idle_add(
                    self._after_inference_callback,
                    _future, # Pass the future to check result/exception
                    Path(settings["output_path"]), # Pass output path
                    settings["auto_play"] # Pass auto_play setting
                )
            )
            self.infer_futures.add(infer_future)

        except Exception as e:
            LOG.exception("Failed to schedule inference:")
            self._show_error_dialog("Error", f"Failed to start inference:\n{e}")
            self.widgets["infer"].set_sensitive(True) # Re-enable button on failure to schedule

    def _after_inference_callback(self, future: ProcessFuture, output_path: Path, auto_play: bool):
        """Callback executed in the main GTK thread after inference is done."""
        self.widgets["infer"].set_sensitive(True) # Re-enable infer button

        try:
            # Check if the task completed successfully (raises exception if not)
            future.result()
            LOG.info(f"Finished inference to {output_path.stem}{output_path.suffix}")

            # Check if output_path is a directory (when input was a directory)
            # If input was a directory, output_path is likely the specified output directory base.
            # We can't auto-play a directory. Maybe list the files produced?
            # For now, only auto-play if the input was a file.
            input_path_str = self.widgets["input_path"].get_text()
            if input_path_str and Path(input_path_str).is_file():
                 if auto_play and output_path.exists() and output_path.is_file():
                    LOG.info(f"Playing output audio {output_path}")
                    # Play audio in a separate process to avoid blocking the GUI
                    if self.audio_play_future and not self.audio_play_future.done():
                         LOG.warning("Skipping auto-play, another audio is already playing.")
                    else:
                        self.audio_play_future = self.process_pool.schedule(play_audio_process, args=(output_path,))
                 elif auto_play:
                     LOG.warning(f"Auto-play requested, but output file not found or is not a file: {output_path}")

        except Exception as e:
            LOG.error("Inference process failed:")
            LOG.exception(e)
            # Error dialog is handled in _check_futures

    def _on_start_vc_clicked(self, button: Gtk.Button):
        """Handler for the Start Voice Changer button."""
        if self.realtime_future and not self.realtime_future.done():
            LOG.warning("Realtime voice changer is already running.")
            self._show_info_dialog("Info", "Voice changer is already running.")
            return

        settings = self._get_current_settings()

        # Perform basic validation relevant to realtime
        if settings["auto_predict_f0"]:
             LOG.warning("Auto Predict F0 must be off for realtime.")
             self._show_error_dialog("Configuration Error", "Auto Predict F0 must be turned off for real-time inference.")
             return

        # Get device indices from the combo boxes
        input_device_name = self.widgets["input_device"].get_active_text()
        output_device_name = self.widgets["output_device"].get_active_text()

        input_device_index = -1
        output_device_index = -1

        try:
             # Find the index corresponding to the selected name
             # Re-get device lists just in case (though _update_devices_combo updates instance vars)
             input_devices_list, output_devices_list, self._input_device_indices, self._output_device_indices = get_devices(update=False)

             if input_device_name in input_devices_list:
                  input_device_index = self._input_device_indices[input_devices_list.index(input_device_name)]
             if output_device_name in output_devices_list:
                  output_device_index = self._output_device_indices[output_devices_list.index(output_device_name)]

             if input_device_index == -1:
                  LOG.error(f"Selected input device not found: {input_device_name}")
                  self._show_error_dialog("Device Error", f"Selected input device not found:\n{input_device_name}\n\nPlease refresh devices.")
                  return
             if output_device_index == -1:
                   LOG.error(f"Selected output device not found: {output_device_name}")
                   self._show_error_dialog("Device Error", f"Selected output device not found:\n{output_device_name}\n\nPlease refresh devices.")
                   return

        except Exception as e:
             LOG.exception("Error getting device indices:")
             self._show_error_dialog("Device Error", f"Failed to determine device indices:\n{e}\n\nPlease refresh devices.")
             return

        # More validation
        if not Path(settings["model_path"]).exists():
             self._show_error_dialog("File Not Found", f"Model path not found:\n{settings['model_path']}")
             return
        if not Path(settings["config_path"]).exists():
             self._show_error_dialog("File Not Found", f"Config path not found:\n{settings['config_path']}")
             return
        if settings["cluster_model_path"] and not Path(settings["cluster_model_path"]).exists():
             self._show_error_dialog("File Not Found", f"Cluster model path not found:\n{settings['cluster_model_path']}")
             return
        if not settings["speaker"]:
             self._show_error_dialog("Configuration Error", "No speaker selected.")
             return
        if self.widgets["speaker"].get_active_text() == "No speakers found" or self.widgets["speaker"].get_active_text() == "Error loading speakers":
             self._show_error_dialog("Configuration Error", "Speaker list is not loaded correctly. Check config path.")
             return


        LOG.info("Starting realtime voice changer...")
        self._update_realtime_buttons(running=True) # Update buttons immediately

        try:
            # Schedule realtime in a separate process
            self.realtime_future = self.process_pool.schedule(
                realtime,
                kwargs=dict(
                    # paths
                    model_path=Path(settings["model_path"]),
                    config_path=Path(settings["config_path"]),
                    speaker=settings["speaker"],
                    # svc config
                    cluster_model_path=(
                        Path(settings["cluster_model_path"])
                        if settings["cluster_model_path"]
                        else None
                    ),
                    transpose=settings["transpose"],
                    auto_predict_f0=settings["auto_predict_f0"], # Should be False based on validation
                    cluster_infer_ratio=settings["cluster_infer_ratio"],
                    noise_scale=settings["noise_scale"],
                    f0_method=settings["f0_method"],
                    # slice config (realtime uses a simplified slicing)
                    db_thresh=settings["silence_threshold"],
                    pad_seconds=settings["pad_seconds"], # This might be used differently in realtime
                    chunk_seconds=settings["chunk_seconds"], # This might be used differently in realtime
                    # realtime config
                    crossfade_seconds=settings["crossfade_seconds"],
                    additional_infer_before_seconds=settings["additional_infer_before_seconds"],
                    additional_infer_after_seconds=settings["additional_infer_after_seconds"],
                    block_seconds=settings["block_seconds"],
                    version=settings["realtime_algorithm"], # Integer value now
                    input_device=input_device_index,
                    output_device=output_device_index,
                    device=get_optimal_device() if settings["use_gpu"] else "cpu",
                    passthrough_original=settings["passthrough_original"],
                ),
            )
            # The completion/error handling for realtime_future is done in _check_futures

        except Exception as e:
            LOG.exception("Failed to schedule realtime voice changer:")
            self._show_error_dialog("Error", f"Failed to start voice changer:\n{e}")
            self._update_realtime_buttons(running=False) # Revert buttons on failure to schedule

    def _on_stop_vc_clicked(self, button: Gtk.Button):
        """Handler for the Stop Voice Changer button."""
        if self.realtime_future and not self.realtime_future.done():
            LOG.info("Stopping realtime voice changer...")
            try:
                # Attempt to cancel the process. This might take a moment.
                self.realtime_future.cancel()
                LOG.info("Cancellation requested.")
                # The actual state update will happen when _check_futures detects it's done/cancelled
            except Exception as e:
                LOG.error(f"Error requesting cancellation: {e}")
                self._show_error_dialog("Error", f"Failed to request cancellation:\n{e}")
        else:
            LOG.warning("Voice changer is not running.")
            self._show_info_dialog("Info", "Voice changer is not running.")
            self._update_realtime_buttons(running=False) # Ensure buttons are correct


    def _update_realtime_buttons(self, running: bool | None = None):
        """Updates the state of the Start/Stop VC buttons."""
        # Determine running state if not explicitly provided
        if running is None:
             running = self.realtime_future is not None and not self.realtime_future.done()

        self.widgets["start_vc"].set_sensitive(not running)
        self.widgets["stop_vc"].set_sensitive(running)


    def _on_refresh_devices_clicked(self, button: Gtk.Button):
        """Handler for the Refresh Devices button."""
        LOG.info("Refreshing audio devices.")
        self._update_devices_combo(update_sd=True)
        self._show_info_dialog("Info", "Audio device list refreshed.")

    def _update_speaker_combo(self) -> None:
        """Updates the speaker combo box based on the config file."""
        config_path_str = self.widgets["config_path"].get_text()
        config_path = Path(config_path_str)
        speaker_combo = self.widgets["speaker"]

        # Clear existing items
        speaker_combo.remove_all()

        if config_path.exists() and config_path.is_file():
            try:
                # Ensure utils is accessible relative to this script's location or via sys.path
                # from . import utils # Keep this if part of a package
                # If running as standalone, ensure so_vits_svc_fork is in sys.path
                # import so_vits_svc_fork.utils as utils # Or this if imported directly

                # Use the imported utils
                hp = utils.get_hparams(config_path_str)
                LOG.debug(f"Loaded config from {config_path_str}")
                speakers = list(hp.__dict__["spk"].keys())
                if speakers:
                    for speaker_name in speakers:
                        speaker_combo.append_text(speaker_name)
                    speaker_combo.set_active(0) # Select the first speaker
                    speaker_combo.set_sensitive(True)
                else:
                    speaker_combo.append_text("No speakers found")
                    speaker_combo.set_active(0)
                    speaker_combo.set_sensitive(False)
                    LOG.warning(f"No speakers found in config file: {config_path_str}")

            except Exception as e:
                LOG.exception(f"Error loading config file {config_path_str}:")
                speaker_combo.append_text("Error loading speakers")
                speaker_combo.set_active(0)
                speaker_combo.set_sensitive(False)
                self._show_error_dialog("Config Error", f"Failed to load speakers from config file:\n{e}\n\nCheck the config path.")

        else:
            LOG.warning(f"Config path does not exist or is not a file: {config_path_str}")
            speaker_combo.append_text("Config not found")
            speaker_combo.set_active(0)
            speaker_combo.set_sensitive(False)


    def _update_devices_combo(self, update_sd: bool = False) -> None:
        """Updates the audio device combo boxes."""
        input_combo = self.widgets["input_device"]
        output_combo = self.widgets["output_device"]

        # Store current selections to try and restore them
        current_input_text = input_combo.get_active_text()
        current_output_text = output_combo.get_active_text()

        # Clear existing items
        input_combo.remove_all()
        output_combo.remove_all()

        try:
            (
                input_devices,
                output_devices,
                self._input_device_indices,
                self._output_device_indices,
            ) = get_devices(update=update_sd)

            for device_name in input_devices:
                input_combo.append_text(device_name)
            for device_name in output_devices:
                output_combo.append_text(device_name)

            # Try to restore the previous selection or set a default
            if current_input_text in input_devices:
                 input_combo.set_active(input_devices.index(current_input_text))
            elif input_devices:
                 # Try to select the sounddevice default input device
                 try:
                      sd_default_in, _ = sd.default.device
                      if sd_default_in in self._input_device_indices:
                          default_idx = self._input_device_indices.index(sd_default_in)
                          input_combo.set_active(default_idx)
                      else:
                           input_combo.set_active(0) # Fallback to first
                 except Exception:
                      input_combo.set_active(0) # Fallback to first
            else:
                 input_combo.append_text("No Input Devices Found")
                 input_combo.set_active(0)
                 input_combo.set_sensitive(False)


            if current_output_text in output_devices:
                 output_combo.set_active(output_devices.index(current_output_text))
            elif output_devices:
                 # Try to select the sounddevice default output device
                 try:
                      _, sd_default_out = sd.default.device
                      if sd_default_out in self._output_device_indices:
                          default_idx = self._output_device_indices.index(sd_default_out)
                          output_combo.set_active(default_idx)
                      else:
                           output_combo.set_active(0) # Fallback to first
                 except Exception:
                       output_combo.set_active(0) # Fallback to first
            else:
                 output_combo.append_text("No Output Devices Found")
                 output_combo.set_active(0)
                 output_combo.set_sensitive(False)


        except Exception as e:
            LOG.exception("Error listing audio devices:")
            input_combo.remove_all() # Clear any partial lists
            output_combo.remove_all()
            input_combo.append_text("Error loading devices")
            output_combo.append_text("Error loading devices")
            input_combo.set_active(0)
            output_combo.set_active(0)
            input_combo.set_sensitive(False)
            output_combo.set_sensitive(False)
            self._show_error_dialog("Device Error", f"Failed to list audio devices:\n{e}")


    def _on_presets_changed(self, combo: Gtk.ComboBoxText):
        """Handler for when the selected preset changes."""
        preset_name = combo.get_active_text()
        if preset_name and preset_name not in ["Error loading presets", "No presets found"]:
            LOG.info(f"Applying preset: {preset_name}")
            self._apply_preset(preset_name)
        elif not preset_name:
            LOG.debug("Preset selection cleared.")
        else:
            LOG.warning(f"Selected preset '{preset_name}' is invalid. Cannot apply.")


    def _apply_preset(self, name: str) -> None:
        """Applies the settings from a named preset."""
        presets = load_presets()
        if name in presets:
            preset_settings = presets[name]
            LOG.debug(f"Applying settings for preset '{name}': {preset_settings}")
            for key, value in preset_settings.items():
                if key in self.widgets:
                    widget = self.widgets[key]
                    try:
                        if isinstance(widget, Gtk.Entry):
                            widget.set_text(str(value))
                        elif isinstance(widget, Gtk.CheckButton):
                            widget.set_active(bool(value))
                        elif isinstance(widget, Gtk.Scale):
                            # Sliders use Adjustments
                             adj = self.sliders.get(key)
                             if adj:
                                 # Ensure value is within the adjustment's bounds
                                 adj_lower = adj.get_lower()
                                 adj_upper = adj.get_upper()
                                 clamped_value = max(adj_lower, min(adj_upper, float(value)))
                                 adj.set_value(clamped_value)
                             else:
                                 LOG.warning(f"No adjustment found for slider key: {key}")
                        elif isinstance(widget, Gtk.ComboBoxText):
                            model = widget.get_model()
                            found = False
                            if model:
                                # Try to find the value in the model
                                for i in range(model.get_n_items()):
                                     item_text = model.get_string(i)
                                     if item_text == str(value):
                                         widget.set_active(i)
                                         found = True
                                         break
                            if not found:
                                LOG.warning(f"Value '{value}' not found in combo box '{key}' options. Keeping current selection.")
                        else:
                            LOG.warning(f"Widget type not handled for key {key}: {type(widget)}")
                    except Exception as e:
                        LOG.error(f"Error applying preset value for key {key}: {value} - {e}")
                        # Optionally show an error, but might be too disruptive during preset apply
            # Update related widgets after applying settings
            self._update_speaker_combo() # Update speaker list based on potential new config path
            self._update_devices_combo() # Update device list
            self._on_auto_predict_f0_toggled(self.widgets["auto_predict_f0"], None) # Update transpose sensitivity


        else:
            LOG.warning(f"Preset '{name}' not found.")


    def _on_add_preset_clicked(self, button: Gtk.Button):
        """Handler for the Add Preset button."""
        preset_name = self.widgets["preset_name"].get_text().strip()
        if not preset_name:
            LOG.warning("Preset name is empty.")
            self._show_info_dialog("Warning", "Please enter a name for the preset.")
            return

        settings = self._get_current_settings()
        presets = add_preset(preset_name, settings)

        # Update the preset combo box model
        self.widgets["presets"].remove_all()
        for name in presets.keys():
            self.widgets["presets"].append_text(name)

        # Select the newly added preset
        preset_names_list = list(presets.keys())
        if preset_name in preset_names_list:
             self.widgets["presets"].set_active(preset_names_list.index(preset_name))

        self.widgets["preset_name"].set_text("") # Clear the input field
        LOG.info(f"Preset '{preset_name}' added.")
        self._show_info_dialog("Success", f"Preset '{preset_name}' added.")


    def _on_delete_preset_clicked(self, button: Gtk.Button):
        """Handler for the Delete Preset button."""
        preset_name = self.widgets["presets"].get_active_text()
        if not preset_name or preset_name in ["Error loading presets", "No presets found"]:
            LOG.warning("No valid preset selected to delete.")
            self._show_info_dialog("Warning", "Please select a preset to delete.")
            return

        # Ask for confirmation
        dialog = Gtk.AlertDialog.new(
            "Confirm Delete",
            f"Are you sure you want to delete the preset '{preset_name}'?",
            ["Cancel", "Delete"],
        )
        dialog.set_modal(True)
        dialog.set_buttons_use_markup(False)
        dialog.choose(self, None, self._on_delete_preset_confirmation_response, preset_name)


    def _on_delete_preset_confirmation_response(self, dialog: Gtk.AlertDialog, result: Gio.AsyncResult, preset_name: str):
        """Callback for the delete preset confirmation dialog."""
        try:
            response_id = dialog.choose_finish(result)
            if response_id == 1: # "Delete" button index (0 is Cancel)
                presets = delete_preset(preset_name)
                LOG.info(f"Preset '{preset_name}' deleted.")

                # Update the preset combo box model
                self.widgets["presets"].remove_all()
                preset_names_list = list(presets.keys())
                for name in preset_names_list:
                     self.widgets["presets"].append_text(name)

                # Select the first preset if list is not empty, otherwise clear
                if preset_names_list:
                     self.widgets["presets"].set_active(0)
                     self._apply_preset(self.widgets["presets"].get_active_text()) # Apply the new first preset
                else:
                     self.widgets["presets"].set_active(-1) # No selection
                     # Optionally clear fields or apply defaults if no presets remain

                self._show_info_dialog("Success", f"Preset '{preset_name}' deleted.")

        except Exception as e:
            LOG.error(f"Error during preset deletion confirmation: {e}")
            self._show_error_dialog("Error", f"An error occurred during preset deletion:\n{e}")


    def _get_current_settings(self) -> dict:
        """Collects current settings from the UI widgets."""
        settings = {}
        for key, widget in self.widgets.items():
            if key.endswith("_adj"): # Skip adjustments, get value from the scale widget
                 continue
            elif isinstance(widget, Gtk.Entry):
                settings[key] = widget.get_text()
            elif isinstance(widget, Gtk.CheckButton):
                settings[key] = widget.get_active()
            elif isinstance(widget, Gtk.Scale):
                 settings[key] = widget.get_value()
            elif isinstance(widget, Gtk.ComboBoxText):
                # Store the text value
                text_value = widget.get_active_text()
                settings[key] = text_value if text_value is not None else "" # Handle None if no selection

                # For realtime algorithm, store the integer number
                if key == "realtime_algorithm":
                    if text_value and text_value[0].isdigit():
                        settings[key] = int(text_value[0])
                    else:
                         settings[key] = 1 # Default if parsing fails
            # Add other widget types if necessary
        return settings

    def _show_info_dialog(self, title: str, message: str):
        """Shows a basic information dialog."""
        dialog = Adw.MessageDialog.new(self, title, message)
        dialog.add_responses("ok", "OK")
        dialog.set_response_enabled("ok", True) # Enable the OK button
        dialog.set_default_response("ok")
        # No need for a callback unless you want to react to the user closing it
        dialog.present()


    def _show_error_dialog(self, title: str, message: str):
        """Shows a basic error dialog."""
        dialog = Adw.MessageDialog.new(self, title, message)
        dialog.add_responses("ok", "OK")
        dialog.set_response_enabled("ok", True) # Enable the OK button
        dialog.set_default_response("ok")
        dialog.set_icon_name("dialog-error-symbolic")
        dialog.present()


    def do_close(self):
        """Handle window closing."""
        LOG.info("Closing application.")
        # Attempt to stop background processes cleanly
        if self.realtime_future and not self.realtime_future.done():
            LOG.info("Canceling realtime voice changer process.")
            self.realtime_future.cancel()
            # Give it a moment to terminate
            time.sleep(0.1)

        if self.audio_play_future and not self.audio_play_future.done():
            LOG.info("Canceling audio playback process.")
            self.audio_play_future.cancel()
             # Give it a moment to terminate
            time.sleep(0.1)

        for future in self.infer_futures:
             if not future.done():
                  LOG.info("Canceling inference process.")
                  future.cancel()
                  # Give it a moment to terminate
                  time.sleep(0.1)


        # Shutdown the process pool
        if self.process_pool:
            LOG.info("Shutting down process pool.")
            self.process_pool.stop()
            self.process_pool.join()
            LOG.info("Process pool shut down.")

        # Call the parent close method
        super().do_close()


class VoiceConverterApplication(Adw.Application):
    def __init__(self):
        super().__init__(application_id="com.example.voiceconverter",
                         flags=Gio.ApplicationFlags.FLAGS_NONE)
        self.window: VoiceConverterWindow | None = None

    def do_activate(self):
        """Activates the application (shows the main window)."""
        if not self.window:
            self.window = VoiceConverterWindow(application=self)
        self.window.present()


def main():
    # Basic logging setup (adjust as needed)
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logging.getLogger("pebble").setLevel(logging.WARNING) # Reduce pebble logging verbosity

    LOG.info(f"Starting Voice Converter GUI v{__version__}")

    app = VoiceConverterApplication()
    app.run(sys.argv)

if __name__ == "__main__":
    # Ensure multiprocessing starts correctly, especially on Windows
    multiprocessing.freeze_support()
    main()