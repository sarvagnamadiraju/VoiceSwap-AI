"""
Quick diagnostic test — run this to find out exactly what is failing.
Usage: py -3.12 test_apis.py
"""
import sys

print("=" * 60)
print("VoiceSwap AI — Diagnostic Test")
print("=" * 60)

# ── Test 1: audio-separator ──
print("\n[1] Testing audio-separator (UVR5)...")
try:
    from audio_separator.separator import Separator
    print("    ✅ audio-separator installed successfully")
except ImportError as e:
    print(f"    ❌ audio-separator NOT installed: {e}")

# ── Test 2: pyworld ──
print("\n[2] Testing pyworld...")
try:
    import pyworld
    print("    ✅ pyworld installed successfully")
except ImportError as e:
    print(f"    ❌ pyworld NOT installed: {e}")

# ── Test 3: librosa / soundfile ──
print("\n[3] Testing librosa and soundfile...")
try:
    import librosa, soundfile
    print("    ✅ librosa + soundfile installed")
except ImportError as e:
    print(f"    ❌ Missing: {e}")

# ── Test 4: gradio_client ──
print("\n[4] Testing gradio_client...")
try:
    from gradio_client import Client
    print("    ✅ gradio_client installed")
except ImportError as e:
    print(f"    ❌ gradio_client NOT installed: {e}")
    sys.exit(1)

# ── Test 5: FreeVC space ──
print("\n[5] Testing HuggingFace FreeVC space (OlaWod/FreeVC)...")
try:
    from gradio_client import Client
    client = Client("OlaWod/FreeVC")
    info = client.view_api(all_endpoints=True)
    print(f"    ✅ FreeVC space is ONLINE!")
    print(f"    API endpoints: {info}")
except Exception as e:
    print(f"    ❌ FreeVC FAILED: {e}")

# ── Test 6: RVC Zero space ──
print("\n[6] Testing HuggingFace RVC-Zero space (r3gm/rvc_zero)...")
try:
    from gradio_client import Client
    client = Client("r3gm/rvc_zero")
    info = client.view_api(all_endpoints=True)
    print(f"    ✅ RVC-Zero space is ONLINE!")
    print(f"    API endpoints: {info}")
except Exception as e:
    print(f"    ❌ RVC-Zero FAILED: {e}")

# ── Test 7: Demucs space ──
print("\n[7] Testing HuggingFace Demucs space (abidlabs/music-separation)...")
try:
    from gradio_client import Client
    client = Client("abidlabs/music-separation")
    print(f"    ✅ Demucs space is ONLINE!")
except Exception as e:
    print(f"    ❌ Demucs FAILED: {e}")

print("\n" + "=" * 60)
print("Diagnostic complete! Copy and paste the results above to your AI assistant.")
print("=" * 60)
