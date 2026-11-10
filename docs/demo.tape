# Reproducible terminal demo, recorded with VHS (https://github.com/charmbracelet/vhs):
#   vhs docs/demo.tape
# Before recording, download the quants once so the run is fast:
#   quantgap score hf://bartowski/Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf --pack packs/Qwen2.5-0.5B-Instruct --quick
Output docs/demo.gif

Set Shell "bash"
Set FontSize 16
Set Width 1200
Set Height 700
Set Padding 20
Set Theme "Catppuccin Mocha"
Set TypingSpeed 40ms

Hide
Type "export PS1='$ '" Enter
Type "clear" Enter
Show

Type "quantgap compare --repo bartowski/Qwen2.5-0.5B-Instruct-GGUF --glob '*Q[48]_*.gguf' --pack packs/Qwen2.5-0.5B-Instruct --budget 1GB --quick --no-damage --chart docs/chart.svg" Enter
Sleep 90s

Type "echo 'Leaderboard: https://saqlain2204.github.io/quantgap'" Enter
Sleep 4s
