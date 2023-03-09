package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"net/http"
	_ "net/http/pprof"
	"os"
	"regexp"
	"strconv"
	"strings"

	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
	"github.com/schollz/progressbar/v3"
	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/decoder"
	"github.com/sugarme/tokenizer/model/wordpiece"
	"github.com/sugarme/tokenizer/normalizer"
	"github.com/sugarme/tokenizer/pretokenizer"
	"github.com/sugarme/tokenizer/processor"
	"github.com/sugarme/tokenizer/util"
	"github.com/sugarme/transformer/bert"
)

type sentimentRow struct {
	post_num  string
	time      string
	text      string
	long      float64
	neutral   float64
	short     float64
	sentiment string
}

type customModel struct {
	tokenizer *tokenizer.Tokenizer
	bertModel *bert.BertForSequenceClassification
}

const (
	fileName        = "test.csv"
	multiprocessNum = 10
	maxLength       = 128
)

var model = loadModel()

func main() {
	go func() {
		log.Println(http.ListenAndServe("localhost:6060", nil))
	}()
	file, _ := os.Open(fileName)
	df := dataframe.ReadCSV(file)
	c := make(chan []sentimentRow)
	splitted := splitDf(df)
	file.Close()
	for _, split := range splitted {
		go splittedDfProcess(split, c)
	}
	var results []sentimentRow
	for i := 0; i < len(splitted); i++ {
		result := <-c
		results = append(results, result...)
	}
	fmt.Println("Saving results...")
	exportToCsv(results)
	fmt.Println("All done!")
}

// channel function
func splittedDfProcess(dataframe dataframe.DataFrame, c chan<- []sentimentRow) {
	var results []sentimentRow
	bar := progressbar.Default(int64(len(dataframe.Col("text").Records())))
	for i := 0; i < dataframe.Nrow(); i++ {
		result := model.bertSentimentProcess(
			dataframe.Subset(
				series.Indexes(int(i)),
			),
		)
		results = append(results, result)
		bar.Add(1)
	}
	c <- results
}

func (m *customModel) bertSentimentProcess(dataframe dataframe.DataFrame) sentimentRow {
	var logit []float64
	ts.NoGrad(func() {
		torchResult, _, _ := m.bertModel.ForwardT(
			processSentenceIntoInput(m.tokenizer, dataframe.Col("text").Records()[0]),
			ts.None,
			ts.None,
			ts.None,
			ts.None,
			false,
		)
		logit = torchResult.MustSoftmax(-1, gotch.Double, true).Float64Values(true)
	})
	var sentiment string
	switch {
	case logit[0] > logit[1] && logit[0] > logit[2]:
		sentiment = "long"
	case logit[1] > logit[0] && logit[1] > logit[2]:
		sentiment = "neutral"
	default:
		sentiment = "short"
	}
	return sentimentRow{
		post_num:  dataframe.Col("post_num").Records()[0],
		time:      dataframe.Col("time").Records()[0],
		text:      dataframe.Col("text").Records()[0],
		long:      logit[0],
		neutral:   logit[1],
		short:     logit[2],
		sentiment: sentiment,
	}
}

func processSentenceIntoInput(tk *tokenizer.Tokenizer, sentence string) *ts.Tensor {
	sentence = strings.ReplaceAll(sentence, "- dc official App", " ")
	sentence = strings.ReplaceAll(sentence, "ㅋ", " ")
	sentence = strings.ReplaceAll(sentence, "\n", " ")
	sentence = strings.ReplaceAll(sentence, "ㅡ", " ")
	reg, _ := regexp.Compile("[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\\-\\%\\?\\.]")
	sentence = reg.ReplaceAllString(sentence, " ")
	words := strings.Split(sentence, " ")
	n := 0
	for _, word := range words {
		if word != "" {
			words[n] = word
			n++
		}
	}
	sentence = strings.Join(words[:n], " ")
	finalEncode, _ := tk.Encode(
		tokenizer.NewSingleEncodeInput(
			tokenizer.NewInputSequence(sentence),
		),
		true,
	)
	switch {
	case finalEncode.Len() > maxLength:
		finalEncode, _ = finalEncode.Truncate(maxLength, 2)
	case finalEncode.Len() < maxLength:
		finalEncode = &tokenizer.PadEncodings(
			[]tokenizer.Encoding{*finalEncode},
			tokenizer.PaddingParams{Strategy: *tokenizer.NewPaddingStrategy(tokenizer.WithFixed(maxLength)),
				Direction: 1,
				PadId:     1,
				PadTypeId: 0,
				PadToken:  "[PAD]",
			},
		)[0]
	default:
	}
	var tokInput = make([]int64, maxLength)
	for i := 0; i < len(finalEncode.Ids); i++ {
		tokInput[i] = int64(finalEncode.Ids[i])
	}
	return ts.MustStack(
		[]ts.Tensor{*ts.TensorFrom(tokInput)},
		0,
	).MustUnsqueeze(int64(0), true)
}

func splitDf(df dataframe.DataFrame) []dataframe.DataFrame {
	len := df.Nrow()
	splitLen := len / multiprocessNum
	if len%multiprocessNum != 0 {
		splitLen++
	}
	splitArray := []series.Indexes{}
	for i := 0; i < multiprocessNum; i++ {
		if (i+1)*splitLen >= len {
			arrayTmp := []int{}
			for j := i * splitLen; j < len; j++ {
				arrayTmp = append(arrayTmp, j)
			}
			indices := series.Indexes(arrayTmp)
			splitArray = append(splitArray, indices)
		} else {
			arrayTmp := []int{}
			for j := i * splitLen; j < (i+1)*splitLen; j++ {
				arrayTmp = append(arrayTmp, j)
			}
			indices := series.Indexes(arrayTmp)
			splitArray = append(splitArray, indices)
		}
	}
	var splitted []dataframe.DataFrame
	for _, indexArray := range splitArray {
		subset := df.Subset(indexArray)
		splitted = append(splitted, subset)
	}
	return splitted
}

func loadModel() *customModel {
	defer fmt.Println("Successfully loaded model")
	return &customModel{
		tokenizer: loadTokenizer(),
		bertModel: loadBertModel(),
	}
}

func loadTokenizer() *tokenizer.Tokenizer {
	currDir, _ := os.Getwd()
	util.CdToThis()
	defer util.CdBack(currDir)
	model, _ := wordpiece.NewWordPieceFromFile("model/tokenizer/vocab.txt", "[UNK]") // load tokenizer on the basis of vocab.txt
	tk := tokenizer.NewTokenizer(model)
	tk.WithNormalizer(normalizer.NewBertNormalizer(true, true, true, true))
	tk.WithPreTokenizer(pretokenizer.NewBertPreTokenizer())
	tk.WithDecoder(decoder.DefaultWordpieceDecoder())
	var specialTokens []tokenizer.AddedToken
	specialTokens = append(specialTokens, tokenizer.NewAddedToken("[MASK]", true))
	tk.AddSpecialTokens(specialTokens)
	sepId, _ := tk.TokenToId("[SEP]")
	clsId, _ := tk.TokenToId("[CLS]")
	sep := processor.PostToken{Id: sepId, Value: "[SEP]"}
	cls := processor.PostToken{Id: clsId, Value: "[CLS]"}
	tk.WithPostProcessor(processor.NewBertProcessing(sep, cls))
	return tk
}

func loadBertModel() *bert.BertForSequenceClassification {
	vs := nn.NewVarStore(gotch.CPU)
	bertConfig, _ := bert.ConfigFromFile("model/bert_config.json") // load bert config file
	bertConfig.Id2Label = map[int64]string{0: "long", 1: "neutral", 2: "short"}
	bertConfig.OutputAttentions = true
	bertConfig.OutputHiddenStates = true
	mymodel := bert.NewBertForSequenceClassification(vs.Root(), bertConfig, false)
	errVarstore := vs.Load("model/sentiment_model.gt") // load model weight
	if errVarstore != nil {
		log.Fatalf("cannot load weights to varstore: %s", errVarstore)
	}
	return mymodel
}

func exportToCsv(results []sentimentRow) {
	saveFile, _ := os.Create("results.csv")
	defer saveFile.Close()
	writer := csv.NewWriter(saveFile)
	defer writer.Flush()
	headers := []string{"post_num", "time", "text", "long", "neutral", "short", "sentiment"}
	errHead := writer.Write(headers)
	if errHead != nil {
		fmt.Println(errHead)
	}
	bar := progressbar.Default(int64(len(results)))
	for i := 0; i < len(results); i++ {
		row := []string{results[i].post_num, results[i].time, results[i].text, strconv.FormatFloat(results[i].long, 'f', 6, 64), strconv.FormatFloat(results[i].neutral, 'f', 6, 64), strconv.FormatFloat(results[i].short, 'f', 6, 64), results[i].sentiment}
		errWrite := writer.Write(row)
		if errWrite != nil {
			fmt.Println(errWrite)
		}
		bar.Add(1)
	}
}
