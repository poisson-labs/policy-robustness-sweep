export type Panel = "top" | "blueprint" | "selection" | "time";
export type PanelState = "hidden" | "collapsed" | "expanded";
export type Backend = "webgpu" | "webgl";
export type VideoDecoder = "auto" | "prefer_software" | "prefer_hardware";
export interface LoginOptions {
    /** URL to redirect to after successful OAuth login (e.g. "/signed-in" or "https://example.com/signed-in"). */
    signed_in_url: string;
    /** URL to redirect to after logout (e.g. "/signed-out" or "https://example.com/signed-out"). */
    signed_out_url: string;
}
export interface WebViewerOptions {
    /** Url to the example manifest. Unused if `hide_welcome_screen` is set to `true`. */
    manifest_url?: string;
    /** The render backend used by the viewer. Either "webgl" or "webgpu". Prefers "webgpu". */
    render_backend?: Backend;
    /** Video decoder config used by the viewer. Either "auto", "prefer_software" or "prefer_hardware". */
    video_decoder?: VideoDecoder;
    /** If set to `true`, hides the welcome screen, which contains our examples. Defaults to `false`. */
    hide_welcome_screen?: boolean;
    /**
     * Allow the viewer to handle fullscreen mode.
     * This option sets canvas style so is not recommended if you are doing anything custom,
     * or are embedding the viewer in an iframe.
     *
     * Defaults to `false`.
     */
    allow_fullscreen?: boolean;
    /**
     * Enable the history feature of the viewer.
     *
     * This is only relevant when `hide_welcome_screen` is `false`,
     * as it's currently only used to allow going between the welcome screen and examples.
     *
     * Defaults to `false`.
     */
    enable_history?: boolean;
    /** The CSS width of the canvas. */
    width?: string;
    /** The CSS height of the canvas. */
    height?: string;
    /** The fallback token to use, if any.
     *
     * The fallback token behaves similarly to the `REDAP_TOKEN` env variable. If set in the
     * enclosing notebook environment, it should be used to set the fallback token.
     */
    fallback_token?: string;
    /**
     * The color theme to use.
     *
     * If not set, the viewer uses the previously persisted theme preference or defaults to "system".
     */
    theme?: "dark" | "light" | "system";
    /**
     * Enable OAuth login in the viewer.
     *
     * When set, the viewer shows login UI and uses the provided URLs for OAuth redirects.
     *
     * To use this:
     * 1. Host the `signed-in.html` and `signed-out.html` pages alongside your viewer.
     *    Templates can be found at:
     *    - https://github.com/rerun-io/rerun/blob/main/crates/viewer/re_web_viewer_server/web_viewer/signed-in.html
     *    - https://github.com/rerun-io/rerun/blob/main/crates/viewer/re_web_viewer_server/web_viewer/signed-out.html
     * 2. Set the URLs to those pages here.
     * 3. Contact your Rerun representative to have the redirect URLs
     *    and origin whitelisted in the OAuth configuration.
     *
     * When not set (default), login UI is hidden. Token-based auth still works.
     */
    login?: LoginOptions;
}
/**
 * The public interface is @see {WebViewerOptions}. This adds a few additional, internal options.
 *
 * @private
 */
export interface AppOptions extends WebViewerOptions {
    /** The url that's used when sharing web viewer urls
     *
     * If not set, the viewer will use the url of the page it is embedded in.
     */
    viewer_base_url?: string;
    /** Whether the viewer is running in a notebook. */
    notebook?: boolean;
    url?: string;
    panel_state_overrides?: Partial<{
        [K in Panel]: PanelState;
    }>;
    on_viewer_event?: (event_json: string) => void;
    fullscreen?: FullscreenOptions;
}
/** An event produced in the Viewer. */
export type ViewerEvent = PlayEvent | PauseEvent | TimeUpdateEvent | TimelineChangeEvent | SelectionChangeEvent | RecordingOpenEvent;
/**
 * Properties available on all {@link ViewerEvent} types.
 */
export type ViewerEventBase = {
    application_id: string;
    recording_id: string;
    partition_id?: string;
};
/**
 * Fired when the timeline starts playing.
 */
export type PlayEvent = ViewerEventBase & {
    type: "play";
};
/**
 * Fired when the timeline stops playing.
 */
export type PauseEvent = ViewerEventBase & {
    type: "pause";
};
/**
 * Fired when the timepoint changes.
 */
export type TimeUpdateEvent = ViewerEventBase & {
    type: "time_update";
    time: number;
};
/**
 * Fired when a different timeline is selected.
 */
export type TimelineChangeEvent = ViewerEventBase & {
    type: "timeline_change";
    timeline: string;
    time: number;
};
/**
 * Fired when the selection changes.
 *
 * This event is fired each time any part of the event payload changes,
 * this includes for example clicking on different parts of the same
 * entity in a 2D or 3D view.
 */
export type SelectionChangeEvent = ViewerEventBase & {
    type: "selection_change";
    items: SelectionChangeItem[];
};
/**
 * Fired when a new recording is opened in the Viewer.
 *
 * For `rrd` file or stream, a recording is considered "open" after
 * enough information about the recording, such as its ID and source,
 * is received.
 *
 * Contains some basic information about the origin of the recording.
 */
export type RecordingOpenEvent = ViewerEventBase & {
    type: "recording_open";
    /**
     * Where the recording came from.
     *
     * The value should be considered unstable, which is why we don't
     * list the possible values here.
     */
    source: string;
    /**
     * Version of the SDK used to create this recording.
     *
     * Uses semver format.
     */
    version?: string;
};
type _GetViewerEvent<K> = Extract<ViewerEvent, {
    type: K;
}>;
type _ViewerEventNames = ViewerEvent["type"];
type ViewerEventMap = {
    [K in _ViewerEventNames]: _GetViewerEvent<K>;
};
/**
 * Selected an entity, or an instance of an entity.
 *
 * If the entity was selected within a view, then this also
 * includes the view's name.
 *
 * If the entity was selected within a 2D or 3D space view,
 * then this also includes the position.
 */
export type EntityItem = {
    type: "entity";
    entity_path: string;
    instance_id?: number;
    view_name?: string;
    position?: [number, number, number];
};
/** Selected a view. */
export type ViewItem = {
    type: "view";
    view_id: string;
    view_name: string;
};
/** Selected a container. */
export type ContainerItem = {
    type: "container";
    container_id: string;
    container_name: string;
};
/** A single item in a selection. */
export type SelectionChangeItem = EntityItem | ViewItem | ContainerItem;
interface FullscreenOptions {
    get_state: () => boolean;
    on_toggle: () => void;
}
export interface WebViewerEvents extends ViewerEventMap {
    fullscreen: boolean;
    ready: void;
}
export type EventsWithValue = {
    [K in keyof WebViewerEvents as WebViewerEvents[K] extends void ? never : K]: WebViewerEvents[K] extends any[] ? WebViewerEvents[K] : [WebViewerEvents[K]];
};
export type EventsWithoutValue = {
    [K in keyof WebViewerEvents as WebViewerEvents[K] extends void ? K : never]: WebViewerEvents[K];
};
/**
 * Rerun Web Viewer
 *
 * ```ts
 * const viewer = new WebViewer();
 * await viewer.start();
 * ```
 *
 * Data may be provided to the Viewer as:
 * - An HTTP file URL, e.g. `viewer.start("https://app.rerun.io/version/0.36.0/examples/dna.rrd")`
 * - A Rerun gRPC URL, e.g. `viewer.start("rerun+http://127.0.0.1:9876/proxy")`
 * - A stream of log messages, via {@link WebViewer.open_channel}.
 *
 * Callbacks may be attached for various events using {@link WebViewer.on}:
 *
 * ```ts
 * viewer.on("time_update", (time) => console.log(`current time: {time}`));
 * ```
 *
 * For the full list of available events, see {@link ViewerEvent}.
 */
export declare class WebViewer {
    #private;
    constructor();
    /**
     * Start the viewer.
     *
     * @param rrd URLs to `.rrd` files or gRPC connections to our SDK.
     * @param parent The element to attach the canvas onto.
     * @param options Web Viewer configuration.
     */
    start(rrd: string | string[] | null, parent: HTMLElement | null, options: WebViewerOptions | null): Promise<void>;
    /** Internal interface */
    private _on_raw_event;
    /**
     * Register an event listener.
     *
     * Returns a function which removes the listener when called.
     *
     * See {@link ViewerEvent} for a full list of available events.
     */
    on<E extends keyof EventsWithValue>(event: E, callback: (...args: EventsWithValue[E]) => void): () => void;
    on<E extends keyof EventsWithoutValue>(event: E, callback: () => void): () => void;
    /**
     * Register an event listener which runs only once.
     *
     * Returns a function which removes the listener when called.
     *
     * See {@link ViewerEvent} for a full list of available events.
     */
    once<E extends keyof EventsWithValue>(event: E, callback: (value: EventsWithValue[E]) => void): () => void;
    once<E extends keyof EventsWithoutValue>(event: E, callback: () => void): () => void;
    /**
     * Unregister an event listener.
     *
     * The event emitter relies on referential equality to store callbacks.
     * The `callback` passed in must be the exact same _instance_ of the function passed in to `on` or `once`.
     *
     * See {@link ViewerEvent} for a full list of available events.
     */
    off<E extends keyof EventsWithValue>(event: E, callback: (value: EventsWithValue[E]) => void): void;
    off<E extends keyof EventsWithoutValue>(event: E, callback: () => void): void;
    /**
     * The underlying canvas element.
     */
    get canvas(): HTMLCanvasElement | null;
    /**
     * Returns `true` if the viewer is ready to connect to data sources.
     */
    get ready(): boolean;
    /**
     * Open a recording.
     *
     * The viewer must have been started via {@link WebViewer.start}.
     *
     * @param rrd URLs to `.rrd` files or gRPC connections to our SDK.
     */
    open(rrd: string | string[]): void;
    /**
     * Close a recording.
     *
     * The viewer must have been started via {@link WebViewer.start}.
     *
     * @param rrd URLs to `.rrd` files or gRPC connections to our SDK.
     */
    close(rrd: string | string[]): void;
    /**
     * Stop the viewer, freeing all associated memory.
     *
     * The same viewer instance may be started multiple times.
     */
    stop(): void;
    /**
     * Opens a new channel for sending log messages.
     *
     * The channel can be used to incrementally push `rrd` chunks into the viewer.
     *
     * @param channel_name used to identify the channel.
     */
    open_channel(channel_name?: string): LogChannel;
    /**
     * Force a panel to a specific state.
     *
     * @param panel which panel to configure
     * @param state which state to force the panel into
     */
    override_panel_state(panel: Panel, state: PanelState | undefined | null): void;
    /**
     * Toggle panel overrides set via `override_panel_state`.
     *
     * @param value set to a specific value. Toggles the previous value if not provided.
     */
    toggle_panel_overrides(value?: boolean | null): void;
    /**
     * Get the active recording id.
     */
    get_active_recording_id(): string | null;
    /**
     * Set the active recording id.
     *
     * This is the same as clicking on the recording in the Viewer's left panel.
     */
    set_active_recording_id(value: string): void;
    /**
     * Get the play state.
     *
     * This always returns `false` if the recording can't be found.
     */
    get_playing(recording_id: string): boolean;
    /**
     * Set the play state.
     *
     * This does nothing if the recording can't be found.
     */
    set_playing(recording_id: string, value: boolean): void;
    /**
     * Get the current time.
     *
     * The interpretation of time depends on what kind of timeline it is:
     *
     * - For time timelines, this is the time in nanoseconds.
     * - For sequence timelines, this is the sequence number.
     *
     * This always returns `0` if the recording or timeline can't be found.
     */
    get_current_time(recording_id: string, timeline: string): number;
    /**
     * Set the current time.
     *
     * Equivalent to clicking on the timeline in the time panel at the specified `time`.
     * The interpretation of `time` depends on what kind of timeline it is:
     *
     * - For time timelines, this is the time in nanoseconds.
     * - For sequence timelines, this is the sequence number.
     *
     * This does nothing if the recording or timeline can't be found.
     */
    set_current_time(recording_id: string, timeline: string, time: number): void;
    /**
     * Get the active timeline.
     *
     * This always returns `null` if the recording can't be found.
     */
    get_active_timeline(recording_id: string): string | null;
    /**
     * Set the active timeline.
     *
     * This does nothing if the recording or timeline can't be found.
     */
    set_active_timeline(recording_id: string, timeline: string): void;
    /**
     * Get the time range for a timeline.
     *
     * This always returns `null` if the recording or timeline can't be found.
     */
    get_time_range(recording_id: string, timeline: string): {
        min: number;
        max: number;
    } | null;
    /**
     * Toggle fullscreen mode.
     *
     * This does nothing if `allow_fullscreen` was not set to `true` when starting the viewer.
     *
     * Fullscreen mode works by updating the underlying `<canvas>` element's `style`:
     * - `position` to `fixed`
     * - width/height/top/left to cover the entire viewport
     *
     * When fullscreen mode is toggled off, the style is restored to its previous values.
     *
     * When fullscreen mode is toggled on, any other instance of the viewer on the page
     * which is already in fullscreen mode is toggled off. This means that it doesn't
     * have to be tracked manually.
     *
     * This functionality can also be directly accessed in the viewer:
     * - The maximize/minimize top panel button
     * - The `Toggle fullscreen` UI command (accessible via the command palette, CTRL+P)
     */
    toggle_fullscreen(): void;
    set_credentials(access_token: string, email: string): void;
}
export declare class LogChannel {
    #private;
    /**
     * @param on_send
     * @param on_close
     * @param get_state
     */
    constructor(on_send: (data: Uint8Array) => void, on_send_table: (data: Uint8Array) => void, on_close: () => void, get_state: () => "ready" | "starting" | "stopped");
    get ready(): boolean;
    /**
     * Send an `rrd` containing log messages to the viewer.
     *
     * Does nothing if `!this.ready`.
     *
     * @param rrd_bytes Is an rrd file stored in a byte array, received via some other side channel.
     */
    send_rrd(rrd_bytes: Uint8Array): void;
    send_table(table_bytes: Uint8Array): void;
    /**
     * Close the channel.
     *
     * Does nothing if `!this.ready`.
     */
    close(): void;
}
export {};
//# sourceMappingURL=index.d.ts.map