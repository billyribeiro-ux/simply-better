<script lang="ts">
	import type { Signal } from '$lib/types';
	import { fmtNum, fmtR, fmtTime, fmtUsd } from '$lib/format';

	let { signals }: { signals: Signal[] } = $props();

	const PAGE_SIZE = 100;

	type SortKey =
		| 'date'
		| 'symbol'
		| 'setup'
		| 'side'
		| 'entry'
		| 'exit'
		| 'exit_reason'
		| 'stop'
		| 'target'
		| 'prob'
		| 'threshold'
		| 'taken'
		| 'outcome'
		| 'net_ev'
		| 'pnl_r'
		| 'pnl_usd'
		| 'mae_r'
		| 'mfe_r';

	const accessors: Record<SortKey, (s: Signal) => string | number | boolean | null> = {
		date: (s) => s.entry_ts,
		symbol: (s) => s.symbol,
		setup: (s) => s.setup,
		side: (s) => s.side,
		entry: (s) => s.entry_px,
		exit: (s) => s.exit_px,
		exit_reason: (s) => s.exit_reason,
		stop: (s) => s.stop_px,
		target: (s) => s.target_px,
		prob: (s) => s.prob,
		threshold: (s) => s.threshold,
		taken: (s) => s.taken,
		outcome: (s) => s.outcome,
		net_ev: (s) => s.net_ev_r ?? null,
		pnl_r: (s) => s.pnl_r,
		pnl_usd: (s) => s.pnl_usd,
		mae_r: (s) => s.mae_r,
		mfe_r: (s) => s.mfe_r
	};

	const columns: { key: SortKey; label: string; numeric: boolean }[] = [
		{ key: 'date', label: 'date', numeric: false },
		{ key: 'symbol', label: 'sym', numeric: false },
		{ key: 'setup', label: 'setup', numeric: false },
		{ key: 'side', label: 'side', numeric: false },
		{ key: 'entry', label: 'entry', numeric: true },
		{ key: 'exit', label: 'exit', numeric: true },
		{ key: 'exit_reason', label: 'why', numeric: false },
		{ key: 'stop', label: 'stop', numeric: true },
		{ key: 'target', label: 'target', numeric: true },
		{ key: 'prob', label: 'prob', numeric: true },
		{ key: 'threshold', label: 'thr', numeric: true },
		{ key: 'taken', label: 'taken', numeric: false },
		{ key: 'outcome', label: 'outcome', numeric: false },
		{ key: 'net_ev', label: 'net EV', numeric: true },
		{ key: 'pnl_r', label: 'P&L R', numeric: true },
		{ key: 'pnl_usd', label: 'P&L $', numeric: true },
		{ key: 'mae_r', label: 'MAE', numeric: true },
		{ key: 'mfe_r', label: 'MFE', numeric: true }
	];

	let sortKey = $state<SortKey>('date');
	let sortDir = $state<1 | -1>(1);
	let symbolFilter = $state('ALL');
	let setupFilter = $state('ALL');
	let takenFilter = $state('ALL');
	let outcomeFilter = $state('ALL');
	let dateFrom = $state('');
	let dateTo = $state('');
	let page = $state(0);

	const symbols = $derived([...new Set(signals.map((s) => s.symbol))].sort());
	const setups = $derived([...new Set(signals.map((s) => s.setup))].sort());

	const filtered = $derived(
		signals.filter(
			(s) =>
				(dateFrom === '' || s.date >= dateFrom) &&
				(dateTo === '' || s.date <= dateTo) &&
				(symbolFilter === 'ALL' || s.symbol === symbolFilter) &&
				(setupFilter === 'ALL' || s.setup === setupFilter) &&
				(takenFilter === 'ALL' || (takenFilter === 'TAKEN') === s.taken) &&
				(outcomeFilter === 'ALL' || s.outcome === outcomeFilter)
		)
	);

	const sorted = $derived.by(() => {
		const get = accessors[sortKey];
		const dir = sortDir;
		return [...filtered].sort((a, b) => {
			const va = get(a);
			const vb = get(b);
			if (va === null && vb === null) return 0;
			if (va === null) return 1; // nulls last, either direction
			if (vb === null) return -1;
			let cmp: number;
			if (typeof va === 'string' && typeof vb === 'string') cmp = va.localeCompare(vb);
			else cmp = Number(va) - Number(vb);
			return cmp * dir;
		});
	});

	const totalPages = $derived(Math.max(1, Math.ceil(sorted.length / PAGE_SIZE)));
	const safePage = $derived(Math.min(page, totalPages - 1));
	const rows = $derived(sorted.slice(safePage * PAGE_SIZE, (safePage + 1) * PAGE_SIZE));
	const takenCount = $derived(signals.filter((s) => s.taken).length);

	function sortBy(key: SortKey): void {
		if (sortKey === key) {
			sortDir = sortDir === 1 ? -1 : 1;
		} else {
			sortKey = key;
			sortDir = 1;
		}
	}

	function ariaSort(key: SortKey): 'ascending' | 'descending' | undefined {
		if (sortKey !== key) return undefined;
		return sortDir === 1 ? 'ascending' : 'descending';
	}
</script>

<div class="signals">
	<div class="filters">
		<div class="group" role="group" aria-label="Filter by period">
			<span class="eyebrow">period</span>
			<input
				class="chip num"
				type="date"
				aria-label="From date"
				bind:value={dateFrom}
				onchange={() => (page = 0)}
			/>
			<span class="dim num">→</span>
			<input
				class="chip num"
				type="date"
				aria-label="To date"
				bind:value={dateTo}
				onchange={() => (page = 0)}
			/>
		</div>
		<div class="group" role="group" aria-label="Filter by symbol">
			<span class="eyebrow">symbol</span>
			{#each ['ALL', ...symbols] as sym (sym)}
				<button
					class="chip num"
					class:active={symbolFilter === sym}
					aria-pressed={symbolFilter === sym}
					onclick={() => {
						symbolFilter = sym;
						page = 0;
					}}>{sym.toLowerCase()}</button
				>
			{/each}
		</div>
		<div class="group" role="group" aria-label="Filter by setup">
			<span class="eyebrow">setup</span>
			{#each ['ALL', ...setups] as su (su)}
				<button
					class="chip num"
					class:active={setupFilter === su}
					aria-pressed={setupFilter === su}
					onclick={() => {
						setupFilter = su;
						page = 0;
					}}>{su.toLowerCase()}</button
				>
			{/each}
		</div>
		<div class="group" role="group" aria-label="Filter by taken">
			<span class="eyebrow">taken</span>
			{#each ['ALL', 'TAKEN', 'SKIPPED'] as tk (tk)}
				<button
					class="chip num"
					class:active={takenFilter === tk}
					aria-pressed={takenFilter === tk}
					onclick={() => {
						takenFilter = tk;
						page = 0;
					}}>{tk.toLowerCase()}</button
				>
			{/each}
		</div>
		<div class="group" role="group" aria-label="Filter by outcome">
			<span class="eyebrow">outcome</span>
			{#each ['ALL', 'WIN', 'LOSS'] as oc (oc)}
				<button
					class="chip num"
					class:active={outcomeFilter === oc}
					aria-pressed={outcomeFilter === oc}
					onclick={() => {
						outcomeFilter = oc;
						page = 0;
					}}>{oc.toLowerCase()}</button
				>
			{/each}
		</div>
	</div>

	<p class="count num">
		{filtered.length} of {signals.length} signals · {takenCount} taken
	</p>

	<div class="scroll">
		<table>
			<thead>
				<tr>
					{#each columns as col (col.key)}
						<th scope="col" class:r={col.numeric} aria-sort={ariaSort(col.key)}>
							<button class="sort num" onclick={() => sortBy(col.key)}>
								{col.label}<span class="arrow"
									>{sortKey === col.key ? (sortDir === 1 ? ' ▲' : ' ▼') : ''}</span
								>
							</button>
						</th>
					{/each}
				</tr>
			</thead>
			<tbody class="num">
				{#each rows as s (s.id)}
					<tr class:skipped={!s.taken}>
						<td>{s.date}</td>
						<td>{s.symbol}</td>
						<td class="dim">{s.setup}</td>
						<td class:loss={s.side === 'SHORT'} class:gain={s.side === 'LONG'}>{s.side}</td>
						<td class="r">{fmtTime(s.entry_ts)} <span class="dim">{fmtNum(s.entry_px, 2)}</span></td>
						<td class="r">{fmtTime(s.exit_ts)} <span class="dim">{fmtNum(s.exit_px, 2)}</span></td>
						<td
							class:gain={s.exit_reason === 'target'}
							class:loss={s.exit_reason === 'stop'}
							class:dim={s.exit_reason === 'eod'}
							>{s.exit_reason === 'target' ? 'tgt' : s.exit_reason}</td
						>
						<td class="r">{fmtNum(s.stop_px, 2)}</td>
						<td class="r">{fmtNum(s.target_px, 2)}</td>
						<td class="r">{fmtNum(s.prob, 3)}</td>
						<td class="r dim">{fmtNum(s.threshold, 3)}</td>
						<td>{s.taken ? 'yes' : 'no'}</td>
						<td class:gain={s.outcome === 'WIN'} class:loss={s.outcome === 'LOSS'}>{s.outcome}</td>
						<td
							class="r"
							class:gain={(s.net_ev_r ?? 0) > 0}
							class:loss={(s.net_ev_r ?? 0) < 0}
							>{s.net_ev_r === undefined ? '—' : fmtR(s.net_ev_r)}</td
						>
						<td class="r" class:gain={s.pnl_r > 0} class:loss={s.pnl_r < 0}>{fmtR(s.pnl_r)}</td>
						<td class="r" class:gain={(s.pnl_usd ?? 0) > 0} class:loss={(s.pnl_usd ?? 0) < 0}>
							{s.pnl_usd === null ? '—' : fmtUsd(s.pnl_usd, 0, true)}
						</td>
						<td class="r dim">{fmtNum(s.mae_r, 2)}</td>
						<td class="r dim">{fmtNum(s.mfe_r, 2)}</td>
					</tr>
				{:else}
					<tr>
						<td class="empty" colspan={columns.length}>No signals match the current filters.</td>
					</tr>
				{/each}
			</tbody>
		</table>
	</div>

	{#if totalPages > 1}
		<nav class="pager num" aria-label="Signals pagination">
			<button class="chip" disabled={safePage === 0} onclick={() => (page = safePage - 1)}>
				prev
			</button>
			<span>page {safePage + 1} / {totalPages}</span>
			<button
				class="chip"
				disabled={safePage >= totalPages - 1}
				onclick={() => (page = safePage + 1)}
			>
				next
			</button>
		</nav>
	{/if}
</div>

<style>
	.signals {
		display: flex;
		flex-direction: column;
		gap: 12px;
	}

	.filters {
		display: flex;
		flex-wrap: wrap;
		gap: 10px 28px;
	}

	.group {
		display: flex;
		align-items: center;
		gap: 6px;
		flex-wrap: wrap;
	}

	.chip {
		font-size: 11px;
		padding: 3px 9px;
		border: 1px solid var(--hairline);
		color: var(--graphite);
		background: var(--paper);
		border-radius: 0;
	}

	input.chip {
		font-family: var(--font-mono);
		color: var(--ink);
	}

	.chip:hover:not(:disabled) {
		background: var(--wash);
	}

	.chip.active {
		background: var(--ink);
		border-color: var(--ink);
		color: var(--paper);
	}

	.chip:disabled {
		opacity: 0.4;
		cursor: default;
	}

	.count {
		font-size: 11.5px;
		color: var(--graphite);
	}

	.scroll {
		overflow-x: auto;
		border-top: 1px solid var(--hairline);
	}

	table {
		width: 100%;
		font-size: 12px;
		min-width: 1080px;
	}

	th {
		padding: 0;
		border-bottom: 1px solid var(--hairline);
		text-align: left;
		white-space: nowrap;
	}

	th.r {
		text-align: right;
	}

	.sort {
		display: block;
		width: 100%;
		padding: 8px 10px 8px 0;
		font-size: 10.5px;
		font-weight: 500;
		letter-spacing: 0.1em;
		text-transform: uppercase;
		color: var(--graphite);
		text-align: inherit;
		white-space: nowrap;
	}

	.sort:hover {
		color: var(--ink);
	}

	.arrow {
		letter-spacing: 0;
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

	td.empty {
		color: var(--graphite);
		padding-block: 18px;
	}

	.pager {
		display: flex;
		align-items: center;
		gap: 14px;
		font-size: 11.5px;
		color: var(--graphite);
	}
</style>
