# Third-party notices

This plugin includes source and a fixed prompt tensor from:

- Project: FlashVSR+, OpenImagingLab
- Repository: https://github.com/OpenImagingLab/FlashVSR_Plus
- Revision: a08fb184a84cc2157c4f271234555fcda7902c87
- License: Apache License 2.0, reproduced in LICENSE
- Changes: internal Python absolute imports are converted to relative imports under the private vendor/runtime namespace. No model, attention, scheduler, preprocessing or decoder mathematics are changed in the vendored Python snapshot.

The optional flashvsr_attn operator is included from the same revision. Its upstream provenance note is preserved at vendor/runtime/flashvsr_attn/OPERATOR_UPSTREAM.md. Original source headers and notices remain in the included files. vendor/manifest.json records hashes and the source revision; scripts/vendor_upstream.py reproduces the import conversion.

Model weights are downloaded separately from ruchangyao/FlashVSR_plus and JunhaoZhuang/FlashVSR-v1.1 on Hugging Face. Their original model cards and terms apply to those assets. The plugin does not bundle DiT, projection or decoder weights.

The node layout references the usability of smthemex/ComfyUI_FlashVSR and lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast. No source from those plugins is included.
