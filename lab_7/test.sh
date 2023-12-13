# search_dir=dataset_1
# for entry in "$search_dir"/*
# do
#     tesseract $entry ${entry%.*} batch.nochop makebox
# done

# function wrap {
#     for i in `seq 0 $1`; do
#         echo “$2$i$3”
#     done
# }

function wrap {
    for name in ${imgs[@]}; do
        echo "${name%.*}$1"
    done

}
imgs=("bad_quality.png" "capcha.png" "commit.png" "don.png" "good_quality.jpg" "google.png" "graph.png" "test.png" "word.png" "zudwa.jpg" "starfall.jpeg")



# # Change this accordingly to number of files, that you want to feed to tesseract or export it as a script parameter.

for name in ${imgs[@]}; do
    tesseract $name ${name%.*} nobatch box.train
done

unicharset_extractor `wrap .box`

echo “ocrb 0 0 1 0 0” > font_properties # tell Tesseract informations about the font

mftraining –F font_properties –U unicharset –O res.unicharset `wrap .tr`

# tesseract -l rus 'good_quality.jpg' 'good_quality' nobatch box.train

# cntraining 'good_quality.tr'

cntraining `wrap .tr`


#  rename all files created by mftraing en cntraining, add the prefix pol.:

mv inttemp res.inttemp
mv normproto res.normproto
mv pffmtable res.pffmtable
mv shapetable res.shapetable

combine_tessdata res.

