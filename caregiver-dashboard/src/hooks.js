import { useEffect, useRef, useState } from "react";

/**
 * Poll an async function. Data is reset whenever `deps` change (e.g. patient switch) so one
 * patient's data is never shown under another. On a failed poll the last good data is kept and
 * `error` is set, so the UI can say "connection problem" without blanking the screen.
 */
export function usePoll(fn, intervalMs, deps = []) {
  const [state, setState] = useState({ data: null, error: null, updatedAt: null, loading: true });
  const fnRef = useRef(fn);
  fnRef.current = fn;

  useEffect(() => {
    let alive = true;
    let timer;
    setState({ data: null, error: null, updatedAt: null, loading: true });

    const run = async () => {
      try {
        const data = await fnRef.current();
        if (alive) setState({ data, error: null, updatedAt: Date.now(), loading: false });
      } catch (error) {
        if (alive) setState((s) => ({ ...s, error, loading: false }));
      }
      if (alive) timer = setTimeout(run, intervalMs);
    };
    run();

    return () => {
      alive = false;
      clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return state;
}

/** Current time in ms, refreshed every `ms` (drives "3 min ago" labels and chart windows). */
export function useNow(ms = 1000) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), ms);
    return () => clearInterval(id);
  }, [ms]);
  return now;
}
