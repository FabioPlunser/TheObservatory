// See https://svelte.dev/docs/kit/types#app.d.ts
// for information about these interfaces
declare global {
	namespace App {
		// interface Error {}
		// interface Locals {}
		// interface PageData {}
		// interface PageState {}
		// interface Platform {}
	}

	type currentPageType = {
		name: string;
		component: any;
	};

	type Company = {
		id: number;
		name: string;
		cloud_url: string;
		cloud_api_key: string;
		license_key?: string;
	};
}

export { };
