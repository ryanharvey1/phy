"""Plugin to add keyboard shortcut for Event PSTH view."""

from phy import IPlugin, connect


class EventPsthShortcutPlugin(IPlugin):
    """Plugin to add keyboard shortcut for the EventPsthView."""

    def attach_to_controller(self, controller):
        @connect
        def on_gui_ready(sender, gui):
            # Add keyboard shortcut to open the Event PSTH view
            @gui.view_actions.add(shortcut="alt+shift+p")
            def show_event_psth():
                """Show or create the Event PSTH view."""
                gui.show_view("EventPsthView")
