import { RefObject, useEffect, useRef } from "react";

/**
 * Stick to bottom while streaming, but immediately yield scroll control
 * whenever the user scrolls up to read earlier content.
 */
export function useSmartAutoScroll(
  containerRef: RefObject<HTMLDivElement | null>,
  deps: unknown[]
) {
  const stickRef = useRef(true);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onScroll = () => {
      const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
      stickRef.current = distance < 88;
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, [containerRef]);

  useEffect(() => {
    const el = containerRef.current;
    if (el && stickRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, deps);

  return {
    scrollToBottom() {
      const el = containerRef.current;
      stickRef.current = true;
      if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    }
  };
}
