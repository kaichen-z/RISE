#!/usr/bin/perl
# pack_bame.pl -- stdlib-only equivalent of tools/pack_frames.py --profile bame
# for machines without numpy (the ORICO host Mac).
#
#   perl pack_bame.pl [--every N] [--delete-mat] <preprocess dir> [...]
#
# Reads every original_mat_<i>_<n>.mat (MAT-v5, one miCOMPRESSED miMATRIX
# 'adcData' of complex double, dims 256 x 16 x 16 x 12, Fortran order), keeps
# doppler bin 0 and writes original_mat_<i>_<n>.npy as complex64 with shape
# (256, 1, 16, 12) in C order -- byte-identical to numpy's
#     np.save(path, cube[:, :1].astype(np.complex64))
# Also writes python_frames.json like pack_frames.py.
# --every N keeps only frames whose index n is a multiple of N (backgrounds: 30).
use strict; use warnings;
use Compress::Zlib ();
use File::Basename qw(basename);

my $every = 1; my $delete = 0; my $outdir; my @dirs;
while (@ARGV) { my $a = shift @ARGV;
  if ($a eq '--every') { $every = shift @ARGV; }
  elsif ($a eq '--out') { $outdir = shift @ARGV; }
  elsif ($a eq '--delete-mat') { $delete = 1; }
  else { push @dirs, $a; } }
die "usage: pack_bame.pl [--every N] [--out DIR] <preprocess dir> ...\n" unless @dirs;
die "--out takes exactly one input dir\n" if $outdir && @dirs != 1;

my ($R, $D, $RX, $TX) = (256, 16, 16, 12);
my $npy_header = make_npy_header("(256, 1, 16, 12)");

for my $dir (@dirs) {
  opendir(my $dh, $dir) or die "$dir: $!"; my @mats = sort grep { /^original_mat_\d+_\d+\.mat$/ } readdir($dh); closedir $dh;
  my ($done, $skipped, $bytes) = (0, 0, 0);
  my $dest = $outdir ? $outdir : $dir;
  if ($outdir) { system('mkdir', '-p', $outdir) == 0 or die "mkdir $outdir failed"; }
  for my $m (@mats) {
    my ($i, $n) = $m =~ /^original_mat_(\d+)_(\d+)\.mat$/;
    if ($every > 1 && $n % $every != 0) { $skipped++; next; }
    (my $out = "$dest/$m") =~ s/\.mat$/.npy/;
    if ($outdir) { my $hyp = "original_hyp_${i}_$n.json"; system('cp', '-p', "$dir/$hyp", "$dest/$hyp") == 0 or die "copy $hyp failed" if -e "$dir/$hyp"; }
    if (-e $out && -s $out == length($npy_header) + $R*$RX*$TX*8) { $done++; $bytes += -s $out; next; }
    my $body = convert("$dir/$m");
    open(my $fh, '>:raw', "$out.tmp") or die "$out: $!"; print $fh $npy_header, $body; close $fh;
    rename("$out.tmp", $out) or die; $done++; $bytes += -s $out;
    unlink("$dir/$m") if $delete;
  }
  open(my $jf, '>', "$dest/python_frames.json") or die; printf $jf qq({"profile": "bame", "dtype": "complex64", "shape": [256, 1, 16, 12], "frames": %d, "every": %d, "packer": "pack_bame.pl"}\n), $done, $every; close $jf;
  printf "%s: %d frames -> .npy (bame), %d skipped by --every %d, %.1f MiB\n", $dir, $done, $skipped, $every, $bytes/1048576;
}

sub convert {
  my ($path) = @_;
  open(my $fh, '<:raw', $path) or die "$path: $!"; local $/; my $buf = <$fh>; close $fh;
  die "$path: not MAT-v5" unless substr($buf, 0, 6) eq 'MATLAB';
  my ($type, $nbytes) = unpack('V V', substr($buf, 128, 8));
  die "$path: top element type $type (expected 15 miCOMPRESSED)" unless $type == 15;
  my $d = Compress::Zlib::uncompress(substr($buf, 136, $nbytes)); die "$path: inflate failed" unless defined $d;
  my ($mt, $mn) = unpack('V V', substr($d, 0, 8)); die "$path: not miMATRIX" unless $mt == 14;
  my $q = 8; my $end = 8 + $mn; my ($flags, @dims, $pr, $pi);
  while ($q < $end) {
    my ($t, $n) = unpack('V V', substr($d, $q, 8)); my ($data, $adv);
    if ($t >> 16) { $n = $t >> 16; $t &= 0xffff; $data = substr($d, $q + 4, $n); $adv = 8; }
    else { $data = ($t == 9) ? undef : substr($d, $q + 8, $n); $adv = 8 + int(($n + 7) / 8) * 8; }
    if ($t == 6 && $n == 8) { ($flags) = unpack('V', $data); }
    elsif ($t == 5) { @dims = unpack('l<*', $data); }
    elsif ($t == 9) { if (!defined $pr) { $pr = [$q + 8, $n]; } else { $pi = [$q + 8, $n]; } }
    $q += $adv;
  }
  die "$path: not complex double" unless defined $flags && ($flags & 0xff) == 6 && ($flags & 0x800);
  die "$path: dims @dims" unless "@dims" eq "$R $D $RX $TX";
  die "$path: missing pr/pi" unless $pr && $pi;
  # doppler bin 0: for (rx, tx) the 256 range values are contiguous at Fortran offset 256*16*(rx + 16*tx)
  my @re_rows; my @im_rows;   # [rx*TX+tx] -> packed 256 floats
  for my $tx (0 .. $TX-1) { for my $rx (0 .. $RX-1) {
    my $off = $R * $D * ($rx + $RX * $tx) * 8;
    my @re = unpack('d<*', substr($d, $pr->[0] + $off, $R * 8));
    my @im = unpack('d<*', substr($d, $pi->[0] + $off, $R * 8));
    $re_rows[$rx * $TX + $tx] = \@re; $im_rows[$rx * $TX + $tx] = \@im;
  } }
  # C order output: index r*(RX*TX) + rx*TX + tx ; each complex64 = float32 re, float32 im
  my $out = '';
  for my $r (0 .. $R-1) { for my $k (0 .. $RX*$TX-1) {
    $out .= pack('f< f<', $re_rows[$k][$r], $im_rows[$k][$r]);
  } }
  return $out;
}

sub make_npy_header {
  my ($shape) = @_;
  my $dict = "{'descr': '<c8', 'fortran_order': False, 'shape': $shape, }";
  my $pre = "\x93NUMPY\x01\x00"; my $len = length($dict) + 1;           # +1 for the newline
  my $pad = (64 - ((length($pre) + 2 + $len) % 64)) % 64; $dict .= ' ' x $pad . "\n";
  return $pre . pack('v', length($dict)) . $dict;
}
