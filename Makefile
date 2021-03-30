PYTHON=python3
OUTPUT=lenet5
EPOCHS=3


all: train_export

train_export:
	${PYTHON} -m webmnist -o ${OUTPUT} -e ${EPOCHS} --train --export

train:
	${PYTHON} -m webmnist -o ${OUTPUT} -e ${EPOCHS} --train

export:
	${PYTHON} -m webmnist -o ${OUTPUT} --export