const dns = require('node:dns').promises;
const http = require('node:http');
const https = require('node:https');
const net = require('node:net');

const MAX_BYTES = 5 * 1024 * 1024;
const REQUEST_TIMEOUT_MS = 8000;

class UrlImageError extends Error {
  constructor(message, status = 400) { super(message); this.status = status; }
}

function pinnedLookup(address, family) {
  return (_hostname, options, callback) => {
    if (typeof options === 'function') { callback = options; options = {}; }
    if (options && options.all) callback(null, [{ address, family }]);
    else callback(null, address, family);
  };
}

function allowedHosts(value = process.env.ALLOWED_IMAGE_HOSTS || '') {
  return new Set(value.split(',').map((host) => host.trim().toLowerCase()).filter(Boolean));
}

function extractImageUrl(body) {
  if (typeof body === 'string') return body.trim();
  if (body && typeof body === 'object') return String(body.url || body.imageUrl || '').trim();
  return '';
}

function isPublicAddress(address) {
  const version = net.isIP(address);
  if (version === 4) {
    const octets = address.split('.').map(Number);
    const [a, b, c] = octets;
    if (a === 0 || a === 10 || a === 127 || a >= 224) return false;
    if (a === 100 && b >= 64 && b <= 127) return false;
    if (a === 169 && b === 254) return false;
    if (a === 172 && b >= 16 && b <= 31) return false;
    if (a === 192 && (b === 168 || (b === 0 && c === 0) || (b === 0 && c === 2) || (b === 88 && c === 99))) return false;
    if (a === 198 && (b === 18 || b === 19 || (b === 51 && c === 100))) return false;
    if (a === 203 && b === 0 && c === 113) return false;
    if (a === 255 && b === 255 && c === 255) return false;
    return true;
  }
  if (version === 6) {
    const ip = address.toLowerCase();
    if (ip === '::' || ip === '::1' || ip.startsWith('fc') || ip.startsWith('fd') || ip.startsWith('fe8') || ip.startsWith('fe9') || ip.startsWith('fea') || ip.startsWith('feb') || ip.startsWith('ff') || ip.startsWith('2001:db8:')) return false;
    const mapped = ip.match(/::ffff:(\d+\.\d+\.\d+\.\d+)$/);
    if (mapped) return isPublicAddress(mapped[1]);
    return true;
  }
  return false;
}

async function fetchAuthorizedImage(value, options = {}) {
  if (typeof value !== 'string' || value.length > 2048) throw new UrlImageError('imageUrl must be a URL under 2048 characters.');
  let url;
  try { url = new URL(value); } catch { throw new UrlImageError('imageUrl is not a valid URL.'); }
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.hash) throw new UrlImageError('Only plain HTTP(S) image URLs without credentials or fragments are supported.');
  const hostname = url.hostname.toLowerCase();
  if (net.isIP(hostname)) throw new UrlImageError('IP-address URLs are not allowed. Use an allowlisted hostname.');
  const hosts = options.hosts || allowedHosts();
  if (!hosts.has(hostname)) throw new UrlImageError(`Host is not allowlisted. Configure ALLOWED_IMAGE_HOSTS for an authorized host (received: ${hostname}).`, 403);

  let resolved;
  try { resolved = await dns.lookup(hostname, { all: true, verbatim: true }); }
  catch { throw new UrlImageError('Image host could not be resolved.', 502); }
  if (resolved.length === 0 || resolved.some((entry) => !isPublicAddress(entry.address))) throw new UrlImageError('Image host resolves to a restricted network address.', 403);
  const pinned = resolved[0];
  const client = url.protocol === 'https:' ? https : http;

  return new Promise((resolve, reject) => {
    let settled = false;
    const fail = (error) => { if (!settled) { settled = true; reject(error); } };
    const request = client.request({
      protocol: url.protocol,
      hostname,
      port: url.port || undefined,
      path: `${url.pathname}${url.search}`,
      method: 'GET',
      headers: { Accept: 'image/*' },
      servername: url.protocol === 'https:' ? hostname : undefined,
      lookup: pinnedLookup(pinned.address, pinned.family)
    }, (response) => {
      if (response.statusCode >= 300 && response.statusCode < 400) {
        response.resume();
        fail(new UrlImageError('Redirects are not followed; use the final image URL.', 400));
        return;
      }
      if (response.statusCode !== 200) {
        response.resume();
        fail(new UrlImageError(`Image host returned HTTP ${response.statusCode}.`, 502));
        return;
      }
      const contentType = String(response.headers['content-type'] || '').split(';')[0].trim().toLowerCase();
      if (!contentType.startsWith('image/')) {
        response.resume();
        fail(new UrlImageError('URL did not return an image content type.', 415));
        return;
      }
      const declaredSize = Number(response.headers['content-length']);
      if (Number.isFinite(declaredSize) && declaredSize > MAX_BYTES) {
        response.destroy();
        fail(new UrlImageError('Remote image exceeds the 5 MB limit.', 413));
        return;
      }
      const chunks = [];
      let size = 0;
      response.on('data', (chunk) => {
        size += chunk.length;
        if (size > MAX_BYTES) {
          response.destroy(new UrlImageError('Remote image exceeds the 5 MB limit.', 413));
          return;
        }
        chunks.push(chunk);
      });
      response.on('end', () => {
        if (settled) return;
        settled = true;
        resolve(Buffer.concat(chunks, size));
      });
      response.on('error', fail);
    });
    request.setTimeout(REQUEST_TIMEOUT_MS, () => request.destroy(new UrlImageError('Remote image request timed out.', 504)));
    request.on('error', fail);
    request.end();
  });
}

module.exports = { UrlImageError, allowedHosts, extractImageUrl, isPublicAddress, pinnedLookup, fetchAuthorizedImage };
