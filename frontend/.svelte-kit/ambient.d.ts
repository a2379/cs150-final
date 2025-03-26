
// this file is generated — do not edit it


/// <reference types="@sveltejs/kit" />

/**
 * Environment variables [loaded by Vite](https://vitejs.dev/guide/env-and-mode.html#env-files) from `.env` files and `process.env`. Like [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private), this module cannot be imported into client-side code. This module only includes variables that _do not_ begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) _and do_ start with [`config.kit.env.privatePrefix`](https://svelte.dev/docs/kit/configuration#env) (if configured).
 * 
 * _Unlike_ [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private), the values exported from this module are statically injected into your bundle at build time, enabling optimisations like dead code elimination.
 * 
 * ```ts
 * import { API_KEY } from '$env/static/private';
 * ```
 * 
 * Note that all environment variables referenced in your code should be declared (for example in an `.env` file), even if they don't have a value until the app is deployed:
 * 
 * ```
 * MY_FEATURE_FLAG=""
 * ```
 * 
 * You can override `.env` values from the command line like so:
 * 
 * ```bash
 * MY_FEATURE_FLAG="enabled" npm run dev
 * ```
 */
declare module '$env/static/private' {
	export const XDG_ACTIVATION_TOKEN: string;
	export const GJS_DEBUG_TOPICS: string;
	export const INSIDE_EMACS: string;
	export const USER: string;
	export const LANGUAGE: string;
	export const npm_config_user_agent: string;
	export const XDG_SEAT: string;
	export const XDG_SESSION_TYPE: string;
	export const npm_node_execpath: string;
	export const SHLVL: string;
	export const XDG_CACHE_HOME: string;
	export const OLDPWD: string;
	export const HOME: string;
	export const DESKTOP_SESSION: string;
	export const GIO_LAUNCHED_DESKTOP_FILE: string;
	export const PAGER: string;
	export const GUILE_LOAD_COMPILED_PATH: string;
	export const XDG_SEAT_PATH: string;
	export const GTK_MODULES: string;
	export const PS1: string;
	export const CINNAMON_VERSION: string;
	export const DBUS_SESSION_BUS_ADDRESS: string;
	export const XDG_STATE_HOME: string;
	export const GIO_LAUNCHED_DESKTOP_FILE_PID: string;
	export const RESTIC_REPOSITORY: string;
	export const INFOPATH: string;
	export const LOGNAME: string;
	export const XDG_SESSION_CLASS: string;
	export const npm_config_registry: string;
	export const COLUMNS: string;
	export const TERM: string;
	export const XDG_SESSION_ID: string;
	export const GNOME_DESKTOP_SESSION_ID: string;
	export const npm_config_node_gyp: string;
	export const PATH: string;
	export const SESSION_MANAGER: string;
	export const GTK3_MODULES: string;
	export const GDM_LANG: string;
	export const npm_package_name: string;
	export const NODE: string;
	export const XDG_RUNTIME_DIR: string;
	export const XDG_SESSION_PATH: string;
	export const npm_config_frozen_lockfile: string;
	export const DISPLAY: string;
	export const DESKTOP_STARTUP_ID: string;
	export const XDG_CURRENT_DESKTOP: string;
	export const DOTNET_BUNDLE_EXTRACT_BASE_DIR: string;
	export const LANG: string;
	export const XAUTHORITY: string;
	export const XDG_SESSION_DESKTOP: string;
	export const XDG_CONFIG_HOME: string;
	export const XDG_DATA_HOME: string;
	export const npm_lifecycle_script: string;
	export const SSH_AUTH_SOCK: string;
	export const XDG_GREETER_DATA_DIR: string;
	export const GUILE_LOAD_PATH: string;
	export const SHELL: string;
	export const npm_package_version: string;
	export const npm_lifecycle_event: string;
	export const npm_config_verify_deps_before_run: string;
	export const NODE_PATH: string;
	export const GDMSESSION: string;
	export const QT_ACCESSIBILITY: string;
	export const XCURSOR_PATH: string;
	export const GJS_DEBUG_OUTPUT: string;
	export const GPG_AGENT_INFO: string;
	export const XDG_VTNR: string;
	export const PWD: string;
	export const npm_execpath: string;
	export const XDG_DATA_DIRS: string;
	export const XDG_CONFIG_DIRS: string;
	export const LINES: string;
	export const npm_command: string;
	export const PNPM_SCRIPT_SRC_DIR: string;
	export const EDITOR: string;
	export const INIT_CWD: string;
}

/**
 * Similar to [`$env/static/private`](https://svelte.dev/docs/kit/$env-static-private), except that it only includes environment variables that begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) (which defaults to `PUBLIC_`), and can therefore safely be exposed to client-side code.
 * 
 * Values are replaced statically at build time.
 * 
 * ```ts
 * import { PUBLIC_BASE_URL } from '$env/static/public';
 * ```
 */
declare module '$env/static/public' {
	
}

/**
 * This module provides access to runtime environment variables, as defined by the platform you're running on. For example if you're using [`adapter-node`](https://github.com/sveltejs/kit/tree/main/packages/adapter-node) (or running [`vite preview`](https://svelte.dev/docs/kit/cli)), this is equivalent to `process.env`. This module only includes variables that _do not_ begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) _and do_ start with [`config.kit.env.privatePrefix`](https://svelte.dev/docs/kit/configuration#env) (if configured).
 * 
 * This module cannot be imported into client-side code.
 * 
 * Dynamic environment variables cannot be used during prerendering.
 * 
 * ```ts
 * import { env } from '$env/dynamic/private';
 * console.log(env.DEPLOYMENT_SPECIFIC_VARIABLE);
 * ```
 * 
 * > In `dev`, `$env/dynamic` always includes environment variables from `.env`. In `prod`, this behavior will depend on your adapter.
 */
declare module '$env/dynamic/private' {
	export const env: {
		XDG_ACTIVATION_TOKEN: string;
		GJS_DEBUG_TOPICS: string;
		INSIDE_EMACS: string;
		USER: string;
		LANGUAGE: string;
		npm_config_user_agent: string;
		XDG_SEAT: string;
		XDG_SESSION_TYPE: string;
		npm_node_execpath: string;
		SHLVL: string;
		XDG_CACHE_HOME: string;
		OLDPWD: string;
		HOME: string;
		DESKTOP_SESSION: string;
		GIO_LAUNCHED_DESKTOP_FILE: string;
		PAGER: string;
		GUILE_LOAD_COMPILED_PATH: string;
		XDG_SEAT_PATH: string;
		GTK_MODULES: string;
		PS1: string;
		CINNAMON_VERSION: string;
		DBUS_SESSION_BUS_ADDRESS: string;
		XDG_STATE_HOME: string;
		GIO_LAUNCHED_DESKTOP_FILE_PID: string;
		RESTIC_REPOSITORY: string;
		INFOPATH: string;
		LOGNAME: string;
		XDG_SESSION_CLASS: string;
		npm_config_registry: string;
		COLUMNS: string;
		TERM: string;
		XDG_SESSION_ID: string;
		GNOME_DESKTOP_SESSION_ID: string;
		npm_config_node_gyp: string;
		PATH: string;
		SESSION_MANAGER: string;
		GTK3_MODULES: string;
		GDM_LANG: string;
		npm_package_name: string;
		NODE: string;
		XDG_RUNTIME_DIR: string;
		XDG_SESSION_PATH: string;
		npm_config_frozen_lockfile: string;
		DISPLAY: string;
		DESKTOP_STARTUP_ID: string;
		XDG_CURRENT_DESKTOP: string;
		DOTNET_BUNDLE_EXTRACT_BASE_DIR: string;
		LANG: string;
		XAUTHORITY: string;
		XDG_SESSION_DESKTOP: string;
		XDG_CONFIG_HOME: string;
		XDG_DATA_HOME: string;
		npm_lifecycle_script: string;
		SSH_AUTH_SOCK: string;
		XDG_GREETER_DATA_DIR: string;
		GUILE_LOAD_PATH: string;
		SHELL: string;
		npm_package_version: string;
		npm_lifecycle_event: string;
		npm_config_verify_deps_before_run: string;
		NODE_PATH: string;
		GDMSESSION: string;
		QT_ACCESSIBILITY: string;
		XCURSOR_PATH: string;
		GJS_DEBUG_OUTPUT: string;
		GPG_AGENT_INFO: string;
		XDG_VTNR: string;
		PWD: string;
		npm_execpath: string;
		XDG_DATA_DIRS: string;
		XDG_CONFIG_DIRS: string;
		LINES: string;
		npm_command: string;
		PNPM_SCRIPT_SRC_DIR: string;
		EDITOR: string;
		INIT_CWD: string;
		[key: `PUBLIC_${string}`]: undefined;
		[key: `${string}`]: string | undefined;
	}
}

/**
 * Similar to [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private), but only includes variables that begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) (which defaults to `PUBLIC_`), and can therefore safely be exposed to client-side code.
 * 
 * Note that public dynamic environment variables must all be sent from the server to the client, causing larger network requests — when possible, use `$env/static/public` instead.
 * 
 * Dynamic environment variables cannot be used during prerendering.
 * 
 * ```ts
 * import { env } from '$env/dynamic/public';
 * console.log(env.PUBLIC_DEPLOYMENT_SPECIFIC_VARIABLE);
 * ```
 */
declare module '$env/dynamic/public' {
	export const env: {
		[key: `PUBLIC_${string}`]: string | undefined;
	}
}
