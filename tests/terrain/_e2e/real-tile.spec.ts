import { describe, it, expect } from 'vitest';
import { decodeQuantizedMesh, decompressGzip } from '../../../src/terrain/QuantizedMeshDecoder';

const REAL_TILE_B64 = 'H4sIAAAAAAAAA7VWa2xUVRCec/fu3n10u0sVSrV2u66PqLHbXZem4c8WeSmhliIY4wN/1EdsRGqi0aQRDBJjIiZE6MON4iOFao0gghqtibAGSTV9maISn40EKqW7XSEEarfXOXPP3d52H7Q23i9n5nTOuTPfzJl7thcPJtreWLv4yNbqUVv17euWNA+6T33SsHIJ0PPUsotZ1n8v/9sn3ddRFVuz79hQTf9Xtecf3ZnYcyZ86PgDfYfjv4Uj+PaEqoahSg0nwxMeKOVQPaoL3KoCVgLNVIVgAhlfkUkCWk00p7/EKtn4XrSYaMhccmhsuS/Ngvs0u1WbiTfoXdol0ztWQ0TSqgv3uLhdxFHE+jReqkP4AhFvkpGcmgPfl8RdE0pSmXCRF9AzSrqwJpB8CKvixqgerBR/XsBRheDV8/B64TqAm7hadf7EG3R/WiRaNaXsk5lNrSGkGGoVUsSKZpe1OqdmIOrlMtQc9PdFZakWOis62Wy8ium0Xal4VgMjqyGeiCi8wmQUldcQq8VrRjWB5BZ1C8z6qehc1Amw7q1t8mqb70PNNvZ1nvno/NOWsStvKF4GsrPX+avnREGpNCwN2PY73rYD/MIeliTnWclqKyrcV3xpAODleKH9vGODxWFOOGoLfmbNBVdb3Kwyr1T6bt7iooV2v6l7fGrkJy9svJCdV6x3pBegbGCVbYdNOc0tT/w0blqfv032mx7BlS/6hqXubm3vgm/rpAPHZp+7/rQxLhkNCaVEWiKLKWWXxCpDm4kkn/MDY3R0DMxgQckHl+l+tL2MpDHWzP0rKBXywZuL0WDUZhJk528SWcjES+fHfShk515sCEb6ctzMxIT9p3yz5Wj06UDpoLkD8lDm0TxP+NfeY4ZzyUeZT7Z8cKF00dyVgY+WO4gq5srdyFMWGWinkKvOZsoBRFUmY1lT58RgHoKR/j/rnLsHND5A+dpE1naqsV5rBtcgGGkGXgT35qXhJf/XouU6BH/3epRlCEZa5zCTHs52ptn6JL03uN15mR5I95+7PkqK39SzM+bFBKvpfhwpfpnON3dcWZygeRqHmcQ1+jFyWIBgpBkUIhhpBlchGOlM/tN73hjL2BslCL5agmM1QiItQTVCwm6txlGDsJC2UCyFus1OPuxTzlSLVYtgpBmsRzDSmXrDeO5zuW+z2Y3fXfp9O/e7zgmMxvQezta3bpRusrmp7iU0LwEPSg/NPfSteomPF3wofXQv+fBbNdPgEsTtlLknjfehk/gBMXNTbKC7qwDBSGf63o05pvd/trsOqM657/+5fEfatyyhlChHiXT2e97o33g3zoSDO3VOeq0ArkA5H8FIz/67W4hgpBkUIfhqEY5bEBJpPRftjgXK2SZOksvcnM2pM2NQjmCkGQQRjDSDCgT3UIHjFYREWoLtCJ7LdhyvIkykM93/+aLD+f3MO0n/PSwQ3cSlsf+N3NJ/W3ch+LwJZRNJgGaUryMYRIhBM7RAK2j/eLei3U4z3gdujOXF3zIf3Ag34fkAypuxjgHMNQRLYTmsgDthI2yCRngeNsNWjNbE/YcwTn105ig/e1ss+GNgOBSrOFMXvWs8NBqKLUqEYuGxlh9eO77jRGQw8ufOv14c2n0ycvKlU2v3bzhYs7e+f/M/939+7we1kaW7Vry5fPeqPbcOBuL+kbKhqnfW7K378vFPa969o+Xu1gfb72lr+PixQw2fPXe4MVo1VHnOP1oep6hxf4zLwIgf/w7E03kF4sH4spFgPBCvPPds18pLwQRaEpuOPP1NY9czPRUjAVw5+n1XR197f3t9tLe980BXRz/N66M970/11fPeR3/UR4MZokziXxr2TwpjDwAA';

describe('real hubei tile e2e', () => {
  it('decodes z12/6553/2707 into valid mesh', async () => {
    const buf = Uint8Array.from(atob(REAL_TILE_B64), (c) => c.charCodeAt(0));
    const raw = await decompressGzip(buf);
    const mesh = await decodeQuantizedMesh(raw, 12, 6553, 2707);
    console.log('verts:', mesh.positions.length / 3, 'indices:', mesh.indices.length / 3);
    console.log('minH/maxH:', mesh.minHeight.toFixed(1), mesh.maxHeight.toFixed(1));
    let rmin = 9, rmax = 0;
    for (let i = 0; i < mesh.positions.length; i += 3) {
      const r = Math.hypot(mesh.positions[i], mesh.positions[i+1], mesh.positions[i+2]);
      rmin = Math.min(rmin, r); rmax = Math.max(rmax, r);
    }
    console.log('radius range:', rmin.toFixed(5), '-', rmax.toFixed(5));
    expect(mesh.positions.length / 3).toBeGreaterThan(50);
    expect(mesh.indices.length / 3).toBeGreaterThan(80);
    expect(mesh.maxHeight).toBeGreaterThan(500);
    expect(rmin).toBeGreaterThan(0.99);   // engine normalisation squashes y/z
    expect(rmax).toBeLessThan(1.001);
  });
});
