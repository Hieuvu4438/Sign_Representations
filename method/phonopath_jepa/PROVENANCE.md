# PLctx fork provenance

Local primitives in `pp/base/` were copied from `method/phonojepa/pj/` on 2026-10-16. Imports were changed to `pp.base`; no dependency on the old method runtime remains. Joint ownership, mirror permutation, state features, PLCM sampling and centered CE retain the original implementation.

The public-anchor fallback in `base/features.py` and `base/window.py` was changed to a constant when shoulders are missing, preventing hidden joints from determining normalization/mirror center. The old whole-clip mean fallback is not used. New preprocessing uses `pp/data.py` to enforce exactly 25 fps and short-gap support; the inherited min(fps,25) sampler is not called.

A0's encoder and predictor have a numerical parity test against the copied PLctx primitives with identical weights. The experiment uses the proposal16 shared preprocessing and augmentation, so A0 must be rerun at the registered 40-epoch budget rather than reusing a historical score.

Source SHA256 at fork time:

```json
{
  "joints.py": "cb57872a5113d76407588d257e4da240213fc5ededabd7bcf3bc88e7b863c338",
  "features.py": "df7fe8a874bbf345c00548dda109067c8e5f8340e8186fa8e78857fa9a4dc7c0",
  "model.py": "6260155147a35101eb4fd6f979975b1447fd684c4c2b3062c685078223aab907",
  "masks.py": "0b7c6f986b55243acfa59a720580849e17b7d7efd1b35208ba2453b24584020e",
  "losses.py": "e62880666d35ce17bca6f8e6f42172476a2e60eb0ce98b9d19632dcddfd492e0",
  "data/window.py": "c668d1cc4c9bc4bb609b924de0bb0027fd18a8dd467a1923a34d07aaa1c64ffa"
}
```
