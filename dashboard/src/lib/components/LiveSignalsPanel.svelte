<script lang="ts">
	import type { LiveFile } from '$lib/types';
	import { fmtNum, fmtTime } from '$lib/format';

	let { live }: { live: LiveFile } = $props();

	const takenCount = $derived(live.signals.filter((s) => s.taken).length);
</script>

<section class="shell" aria-label="Live signals">
	<div class="head">
		<h2 class="eyebrow">Live signals</h2>
		<p class="meta num">
			session {live.session_date} · as of {fmtTime(live.as_of_et)} ET · model through
			{live.trained_through} · base thr {fmtNum(live.threshold_base, 3)}
		</p>
	</div>

	{#if live.signals.length}
		<p class="count num">{live.signals.length} triggers · {takenCount} taken</p>
		<div class="scroll">
			<table>
				<thead>
					<tr>
						<th scope="col">sym</th>
						<th scope="col">setup</th>
						<th scope="col">side</th>
						<th scope="col">trigger</th>
						<th scope="col" class="r">prob</th>
						<th scope="col" class="r">thr</th>
						<th scope="col">anchor</th>
						<th scope="col" class="r">entry stop-order</th>
						<th scope="col" class="r">stop</th>
						<th scope="col" class="r">target</th>
						<th scope="col" class="r">shares</th>
						<th scope="col">status</th>
						<th scope="col">decision</th>
					</tr>
				</thead>
				<tbody class="num">
					{#each live.signals as s (s.id)}
						<tr class:skipped={!s.taken}>
							<td>{s.symbol}</td>
							<td class="dim">{s.setup}</td>
							<td class:loss={s.side === 'SHORT'} class:gain={s.side === 'LONG'}>{s.side}</td>
							<td>{fmtTime(s.trigger_ts)}</td>
							<td class="r">{fmtNum(s.prob, 3)}</td>
							<td class="r dim">{fmtNum(s.threshold, 3)}</td>
							<td class="dim">{s.anchor === 'atr' ? 'ATR ×' : s.anchor}</td>
							<td class="r">
								{fmtNum(s.entry_trigger_px, 2)}
								{#if s.entry_px !== null}
									<span class="dim">→ {fmtNum(s.entry_px, 2)}</span>
								{/if}
							</td>
							<td class="r loss">{fmtNum(s.stop_px, 2)}</td>
							<td class="r gain">{fmtNum(s.target_px, 2)}</td>
							<td class="r">{s.shares}</td>
							<td>
								<span
									class="status"
									class:confirmed={s.status === 'confirmed'}
									class:expired={s.status === 'expired'}>{s.status}</span
								>
							</td>
							<td class:gain={s.taken}>{s.taken ? 'TAKE' : s.tradable ? 'skip' : 'no room'}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{:else}
		<p class="empty num">{live.note ?? 'No triggers so far this session.'}</p>
	{/if}
</section>

<style>
	section {
		margin-top: 28px;
		padding-bottom: 24px;
		border-bottom: 1px solid var(--hairline);
	}

	.head {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		gap: 16px;
		flex-wrap: wrap;
		margin-bottom: 10px;
	}

	.meta {
		font-size: 11.5px;
		color: var(--graphite);
	}

	.count {
		font-size: 11.5px;
		color: var(--graphite);
		margin-bottom: 8px;
	}

	.scroll {
		overflow-x: auto;
		border-top: 1px solid var(--hairline);
	}

	table {
		width: 100%;
		font-size: 12px;
		min-width: 960px;
	}

	th {
		font-family: var(--font-mono);
		font-size: 10.5px;
		font-weight: 500;
		letter-spacing: 0.1em;
		text-transform: uppercase;
		color: var(--graphite);
		text-align: left;
		padding: 8px 10px 8px 0;
		border-bottom: 1px solid var(--hairline);
		white-space: nowrap;
	}

	th.r {
		text-align: right;
	}

	td {
		padding: 6px 10px 6px 0;
		border-bottom: 1px solid var(--hairline);
		white-space: nowrap;
	}

	td.r {
		text-align: right;
	}

	tbody tr:hover {
		background: var(--wash);
	}

	tr.skipped {
		opacity: 0.55;
	}

	.dim {
		color: var(--graphite);
	}

	.status {
		color: var(--graphite);
	}

	.status.confirmed {
		color: var(--ink);
		font-weight: 600;
	}

	.status.expired {
		opacity: 0.6;
	}

	.empty {
		color: var(--graphite);
		font-size: 12.5px;
	}
</style>
