
declare namespace wasm_bindgen {
    /* tslint:disable */
    /* eslint-disable */
    /**
     * The `ReadableStreamType` enum.
     *
     * *This API requires the following crate features to be activated: `ReadableStreamType`*
     */

    export type ReadableStreamType = "bytes";

    export class IntoUnderlyingByteSource {
        private constructor();
        free(): void;
        [Symbol.dispose](): void;
        cancel(): void;
        pull(controller: ReadableByteStreamController): Promise<any>;
        start(controller: ReadableByteStreamController): void;
        readonly autoAllocateChunkSize: number;
        readonly type: ReadableStreamType;
    }

    export class IntoUnderlyingSink {
        private constructor();
        free(): void;
        [Symbol.dispose](): void;
        abort(reason: any): Promise<any>;
        close(): Promise<any>;
        write(chunk: any): Promise<any>;
    }

    export class IntoUnderlyingSource {
        private constructor();
        free(): void;
        [Symbol.dispose](): void;
        cancel(): void;
        pull(controller: ReadableStreamDefaultController): Promise<any>;
    }

    export class WebHandle {
        free(): void;
        [Symbol.dispose](): void;
        /**
         * Add a new receiver streaming data from the given url.
         *
         * Websocket streams are always opened in `Following` mode.
         *
         * It is an error to open a channel twice with the same id.
         */
        add_receiver(url: string): void;
        /**
         * Close an existing channel for streaming data.
         *
         * No-op if the channel is already closed.
         */
        close_channel(id: string): void;
        destroy(): void;
        get_active_recording_id(): string | undefined;
        get_active_timeline(recording_id: string): string | undefined;
        get_playing(recording_id: string): boolean | undefined;
        get_time_for_timeline(recording_id: string, timeline_name: string): number | undefined;
        get_timeline_time_range(recording_id: string, timeline_name: string): any;
        has_panicked(): boolean;
        constructor(app_options: any);
        /**
         * Open a new channel for streaming data.
         *
         * It is an error to open a channel twice with the same id.
         */
        open_channel(id: string, channel_name: string): void;
        override_panel_state(panel: string, state?: string | null): void;
        panic_callstack(): string | undefined;
        panic_message(): string | undefined;
        remove_receiver(url: string): void;
        /**
         * Add an rrd to the viewer directly from a byte array.
         */
        send_rrd_to_channel(id: string, data: Uint8Array): void;
        send_table_to_channel(id: string, data: Uint8Array): void;
        set_active_recording_id(recording_id: string): void;
        /**
         * Set the active timeline.
         *
         * This does nothing if the timeline can't be found.
         */
        set_active_timeline(recording_id: string, timeline_name: string): void;
        set_credentials(access_token: string, email: string): void;
        set_playing(recording_id: string, value: boolean): void;
        set_time_for_timeline(recording_id: string, timeline_name: string, time: number): void;
        start(canvas: any): Promise<void>;
        toggle_panel_overrides(value?: boolean | null): void;
    }

}
declare type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

declare interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_webhandle_free: (a: number, b: number) => void;
    readonly webhandle_add_receiver: (a: number, b: number, c: number) => void;
    readonly webhandle_close_channel: (a: number, b: number, c: number) => void;
    readonly webhandle_destroy: (a: number) => void;
    readonly webhandle_get_active_recording_id: (a: number) => [number, number];
    readonly webhandle_get_active_timeline: (a: number, b: number, c: number) => [number, number];
    readonly webhandle_get_playing: (a: number, b: number, c: number) => number;
    readonly webhandle_get_time_for_timeline: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly webhandle_get_timeline_time_range: (a: number, b: number, c: number, d: number, e: number) => any;
    readonly webhandle_has_panicked: (a: number) => number;
    readonly webhandle_new: (a: any) => [number, number, number];
    readonly webhandle_open_channel: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly webhandle_override_panel_state: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly webhandle_panic_callstack: (a: number) => [number, number];
    readonly webhandle_panic_message: (a: number) => [number, number];
    readonly webhandle_remove_receiver: (a: number, b: number, c: number) => void;
    readonly webhandle_send_rrd_to_channel: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly webhandle_send_table_to_channel: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly webhandle_set_active_recording_id: (a: number, b: number, c: number) => void;
    readonly webhandle_set_active_timeline: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly webhandle_set_credentials: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly webhandle_set_playing: (a: number, b: number, c: number, d: number) => void;
    readonly webhandle_set_time_for_timeline: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly webhandle_start: (a: number, b: any) => any;
    readonly webhandle_toggle_panel_overrides: (a: number, b: number) => void;
    readonly rust_lz4_wasm_shim_calloc: (a: number, b: number) => number;
    readonly rust_lz4_wasm_shim_free: (a: number) => void;
    readonly rust_lz4_wasm_shim_malloc: (a: number) => number;
    readonly rust_lz4_wasm_shim_memcmp: (a: number, b: number, c: number) => number;
    readonly rust_lz4_wasm_shim_memcpy: (a: number, b: number, c: number) => number;
    readonly rust_lz4_wasm_shim_memmove: (a: number, b: number, c: number) => number;
    readonly rust_lz4_wasm_shim_memset: (a: number, b: number, c: number) => number;
    readonly rust_zstd_wasm_shim_calloc: (a: number, b: number) => number;
    readonly rust_zstd_wasm_shim_free: (a: number) => void;
    readonly rust_zstd_wasm_shim_malloc: (a: number) => number;
    readonly rust_zstd_wasm_shim_memcmp: (a: number, b: number, c: number) => number;
    readonly rust_zstd_wasm_shim_memcpy: (a: number, b: number, c: number) => number;
    readonly rust_zstd_wasm_shim_memmove: (a: number, b: number, c: number) => number;
    readonly rust_zstd_wasm_shim_memset: (a: number, b: number, c: number) => number;
    readonly rust_zstd_wasm_shim_qsort: (a: number, b: number, c: number, d: number) => void;
    readonly __wbg_intounderlyingbytesource_free: (a: number, b: number) => void;
    readonly __wbg_intounderlyingsink_free: (a: number, b: number) => void;
    readonly __wbg_intounderlyingsource_free: (a: number, b: number) => void;
    readonly intounderlyingbytesource_autoAllocateChunkSize: (a: number) => number;
    readonly intounderlyingbytesource_cancel: (a: number) => void;
    readonly intounderlyingbytesource_pull: (a: number, b: any) => any;
    readonly intounderlyingbytesource_start: (a: number, b: any) => void;
    readonly intounderlyingbytesource_type: (a: number) => number;
    readonly intounderlyingsink_abort: (a: number, b: any) => any;
    readonly intounderlyingsink_close: (a: number) => any;
    readonly intounderlyingsink_write: (a: number, b: any) => any;
    readonly intounderlyingsource_cancel: (a: number) => void;
    readonly intounderlyingsource_pull: (a: number, b: any) => any;
    readonly wasm_bindgen__convert__closures_____invoke__ha28703b0fc0ac5f5: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen__convert__closures_____invoke__h17feb392561402d4: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen__convert__closures_____invoke__h0b81c5800b6cd5e2: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen__convert__closures_____invoke__h17feb392561402d4_10: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen__convert__closures_____invoke__h17feb392561402d4_11: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen__convert__closures_____invoke__h64d340b52b512fdd: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen__convert__closures_____invoke__h4a090e5af75dc439: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h72e0675ea71ceaf9: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h5f7e507a11f05fb1: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__hde9aedb09f31094c: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h5f7e507a11f05fb1_4: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__ha366fcce789d0db1: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__hd0e2805b66747040: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__hde9aedb09f31094c_9: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h9a9356fa8d11d697: (a: number, b: number) => [number, number];
    readonly wasm_bindgen__convert__closures_____invoke__h0e2ab714131e0149: (a: number, b: number) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h2a38a854c15eed3d: (a: number, b: number) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h46084f6dced1097a: (a: number, b: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_destroy_closure: (a: number, b: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
declare function wasm_bindgen (module_or_path: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;

export type WebHandle = wasm_bindgen.WebHandle;
export default function(): wasm_bindgen;
