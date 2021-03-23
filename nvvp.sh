sudo archlinux-java set java-8-openjdk/jre
nvprof -o prof.nvvp $1
nvvp prof.nvvp
sudo archlinux-java set java-15-openjdk/jre
