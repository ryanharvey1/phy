# Event PSTH View for Phy

This provides an Event PSTH (Peri-Stimulus Time Histogram) view that integrates directly into phy's core view system.

## Files

### Core View (in phy/cluster/views/)
- `psth.py` - The EventPsthView class (integrated into phy core)

### Plugin Files (in plugins/)
- `EventPsthShortcut.py` - Plugin that adds keyboard shortcut for the view
- `EventPsth.py` - Original standalone plugin (legacy, kept for reference)
- `create_test_events.py` - Script to create test event files

## Installation

The EventPsthView is now integrated into phy's core views. You have two options:

### Option 1: Core Integration (Recommended)
The view is automatically available in phy if you've added the files to the core:

1. Copy `psth.py` to `phy/cluster/views/`
2. Update `phy/cluster/views/__init__.py` to import EventPsthView
3. Update `phy/apps/base.py` to include the view creator
4. The view will appear in the View menu automatically

### Option 2: Plugin Shortcut
For easy access without modifying core files:

1. Copy `EventPsthShortcut.py` to your phy plugins directory
2. The view will be available via `Alt+Shift+P` shortcut

## Features

The EventPsthView provides:

1. **Integrated View Window**: Appears as a native phy view that can be docked and managed
2. **Interactive Parameter Control**: Change bin size and time window with mouse wheel
3. **Multi-cluster Support**: Shows PSTH for all selected clusters
4. **Consistent Aesthetics**: Matches phy's visual style
5. **Persistent Settings**: Parameters are saved with phy sessions
6. **Automatic Event Loading**: Finds `.mat` files with "events" in filename

## Usage

### Opening the View

**If using core integration:**
- **View Menu**: View → EventPsthView
- **Keyboard**: Use the standard phy view shortcuts

**If using plugin shortcut:**
- **Keyboard**: `Alt+Shift+P`

### Controls

**Mouse Wheel:**
- `Alt + Wheel`: Adjust bin size (1-100ms)
- `Ctrl + Wheel`: Adjust time window (0.1-10s)

**Text Commands:**
- `pb <ms>`: Set bin width (e.g., `pb 5` for 5ms)
- `pw <start>,<end>`: Set window (e.g., `pw -2,2` for ±2s)
- `pr`: Reload events from file

### Event File Format

Place a `.mat` file with "events" in the filename in your working directory:

```matlab
events.timestamps = [time1; time2; time3; ...];  % Event times in seconds
```

## Architecture

### Core Integration Benefits

By integrating EventPsthView into phy's core views:

1. **Native Integration**: Behaves exactly like built-in views
2. **Standard UI**: Follows all phy view conventions
3. **Performance**: Uses phy's optimized rendering pipeline
4. **State Management**: Automatic saving/loading of view settings
5. **Plugin Compatibility**: Works with all phy plugins

### View Hierarchy

```
ManualClusteringView (base)
├── ScalingMixin (interaction)
└── EventPsthView (implementation)
    ├── HistogramVisual (rendering)
    ├── LineVisual (zero line)
    └── TextVisual (info display)
```

## File Organization

```
phy/
├── cluster/views/
│   ├── __init__.py          # (modified to import EventPsthView)
│   ├── psth.py             # EventPsthView class
│   └── ...
├── apps/
│   ├── base.py             # (modified to include view creator)
│   └── template/gui.py     # (modified to include in default views)
└── ...

plugins/                    # Optional shortcut plugin
├── EventPsthShortcut.py   # Keyboard shortcut plugin
├── EventPsth.py           # Legacy standalone version
└── create_test_events.py  # Test data generator
```

## Migration Guide

### From Plugin to Core View

If migrating from the plugin version:

1. **Remove old plugin**: Delete EventPsthView.py from plugins/
2. **Install core files**: Add psth.py to phy/cluster/views/
3. **Update imports**: Modify __init__.py and base.py as shown
4. **Optional shortcut**: Add EventPsthShortcut.py for easy access

### Backwards Compatibility

- Old EventPsth.py plugin still works independently
- Settings and usage patterns remain the same
- Event file format unchanged

## Performance

- **Cluster Limit**: Max 10 clusters for responsiveness
- **Efficient Computation**: Optimized cross-correlation algorithms
- **GPU Rendering**: Uses OpenGL for smooth visualization
- **Caching**: Leverages phy's spike data caching

## Troubleshooting

**View not available:**
- Check that psth.py is in phy/cluster/views/
- Verify __init__.py imports EventPsthView
- Ensure base.py includes create_event_psth_view

**Shortcut not working:**
- Check EventPsthShortcut.py is in plugins directory
- Verify no conflicts with existing shortcuts
- Try accessing via View menu instead

**No events loaded:**
- Ensure .mat file has "events" in filename
- Check file contains "timestamps" field or direct array
- Verify file is in current working directory
