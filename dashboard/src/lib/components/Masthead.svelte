<script lang="ts">
	import type { Summary } from '$lib/types';
	import { fmtGeneratedAt } from '$lib/format';

	let { summary }: { summary: Summary } = $props();
</script>

<header class="masthead shell">
	<div class="title">
		<p class="eyebrow">Market intelligence engine</p>
		<h1>HOD/LOD Research Terminal</h1>
	</div>
	<dl class="meta num">
		<div>
			<dt>run</dt>
			<dd>{summary.run_id}</dd>
		</div>
		<div>
			<dt>range</dt>
			<dd>{summary.from} → {summary.to}</dd>
		</div>
		<div>
			<dt>generated</dt>
			<dd>{fmtGeneratedAt(summary.generated_at)}</dd>
		</div>
	</dl>
</header>

<style>
	.masthead {
		display: flex;
		justify-content: space-between;
		align-items: flex-end;
		gap: 24px;
		padding-block: 36px 28px;
		flex-wrap: wrap;
	}

	h1 {
		font-family: var(--font-display);
		font-size: clamp(28px, 4.4vw, 44px);
		font-weight: 700;
		font-variation-settings: 'wght' 700;
		letter-spacing: -0.022em;
		line-height: 1.05;
		margin-top: 10px;
	}

	.meta {
		display: grid;
		gap: 3px;
		font-size: 12px;
		color: var(--graphite);
	}

	.meta div {
		display: flex;
		gap: 10px;
		justify-content: flex-end;
	}

	.meta dt {
		color: var(--graphite);
	}

	.meta dd {
		color: var(--ink);
	}

	@media (max-width: 640px) {
		.masthead {
			align-items: flex-start;
			flex-direction: column;
		}

		.meta div {
			justify-content: flex-start;
		}
	}
</style>
