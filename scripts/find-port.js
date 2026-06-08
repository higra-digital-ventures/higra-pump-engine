/**
 * Port utilities — pick a free TCP port, falling back to the next one
 * when the preferred port is already in use.
 *
 * Used by the HPE startup scripts so a busy 8000 (backend) or 3000
 * (frontend) does not block bringing the app up.
 */

const net = require("net");

/** Resolve true if `port` is free to bind on, false otherwise. */
function isFree(port, host = "0.0.0.0") {
  return new Promise((resolve) => {
    const srv = net.createServer();
    srv.once("error", () => resolve(false));
    srv.once("listening", () => srv.close(() => resolve(true)));
    srv.listen(port, host);
  });
}

/**
 * Return the first free port at or after `preferred`.
 * Scans `preferred .. preferred + maxTries - 1`.
 */
async function findFreePort(preferred, maxTries = 50) {
  for (let p = preferred; p < preferred + maxTries; p++) {
    // eslint-disable-next-line no-await-in-loop
    if (await isFree(p)) return p;
  }
  throw new Error(
    `No free port found in range ${preferred}-${preferred + maxTries - 1}`
  );
}

module.exports = { isFree, findFreePort };
