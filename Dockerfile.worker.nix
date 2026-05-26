FROM nixos/nix:2.24.11

ENV DEEPSKY_VERIFY_MODE=container
ENV DEEPSKY_REQUIRE_CUDA=0
ENV USE_XVFB=1
ENV PYTHONUNBUFFERED=1

RUN nix-channel --update
RUN nix-env -iA \
    nixpkgs.python311 \
    nixpkgs.python311Packages.pip \
    nixpkgs.xvfb-run \
    nixpkgs.siril \
    nixpkgs.cfitsio \
    nixpkgs.libtiff \
    nixpkgs.libjpeg \
    nixpkgs.libpng \
    nixpkgs.mesa

WORKDIR /app

COPY deepsky_processor/requirements.txt /tmp/deepsky-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/deepsky-requirements.txt

CMD ["python", "-m", "deepsky_processor.pipeline.main_pipeline", "--doctor", "--mode", "container"]
