import logging
import pandas as pd
import numpy as np
from app.config import config

# Sklearn libraries
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import *
from joblib import dump
from sklearn.tree._tree import TREE_LEAF

# Plotly Libraries
import plotly.graph_objs as go
import plotly.offline as offline
import plotly.figure_factory as ff
# Other Graphing libraries
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil.relativedelta import *
from datetime import date
import calendar

# Visualizing Tree (Does not always work)
# from sklearn import tree
# import pydotplus
# from IPython.display import Image
# from datetime import datetime as dt

sns.set('paper')


class GeneralUtilClass:

    def __init__(self):
        self.logger = logging.getLogger(f'{config.LOGGER_NAME}')

    def months_end_shift(self, dt_date, month_shift):
        self.logger.info("Shifting Date {} months to month end".format(month_shift))
        date_shifted = dt_date + relativedelta(months=month_shift)
        month_end = calendar.monthrange(date_shifted.year, date_shifted.month)[1]
        date_shift_out = date(year=date_shifted.year, month=date_shifted.month, day=month_end)
        return date_shift_out

    def baseline_models(self, classifiers, x_train, y_train, x_test, y_test):
        """
        Include parameters, and what what
        Get baseline parameter estimates and outputs of model.
        """
        self.logger.info("Running BaselineModels function")
        cols = ['classifiers', 'fpr_train', 'tpr_train', 'auc_train',
                'fpr_test', 'tpr_test', 'auc_test']
        df_results = pd.DataFrame(columns=cols)
        model = None
        pred_test = None
        pred_train = None
        for cls in classifiers:
            model = cls.fit(x_train, y_train)
            pred_test = model.predict_proba(x_test)[::, 1]
            # surely there's an easier way to get training predictions
            pred_train = model.predict_proba(x_train)[::, 1]

            fpr_test, tpr_test, _ = roc_curve(y_test,  pred_test)
            auc_test = roc_auc_score(y_test, pred_test)

            fpr_train, tpr_train, _ = roc_curve(y_train,  pred_train)
            auc_train = roc_auc_score(y_train, pred_train)

            df_results = df_results.append({'classifiers': cls.__class__.__name__,
                                            'fpr_train': fpr_train,
                                            'tpr_train': tpr_train,
                                            'auc_train': auc_train,
                                            'fpr_test': fpr_test,
                                            'tpr_test': tpr_test,
                                            'auc_test': auc_test}, ignore_index=True)
        df_results.set_index('classifiers', inplace=True)

        return model, pred_test, pred_train, df_results

    def baseline_model_metrics(self, classifier, x_train, y_train, x_test, y_test):
        """
        Include parameters, and what what
        Get baseline parameter estimates and outputs of model.
        """
        self.logger.info("Running BaselineModels function")
        cols = ['classifiers', 'fpr_train', 'tpr_train', 'auc_train',
                'fpr_test', 'tpr_test', 'auc_test']
        df_results = pd.DataFrame(columns=cols)
        pred_test = classifier.predict_proba(x_test)[::, 1]
        # surely there's an easier way to get training predictions
        pred_train = classifier.predict_proba(x_train)[::, 1]

        fpr_test, tpr_test, _ = roc_curve(y_test,  pred_test)
        auc_test = roc_auc_score(y_test, pred_test)

        fpr_train, tpr_train, _ = roc_curve(y_train,  pred_train)
        auc_train = roc_auc_score(y_train, pred_train)

        df_results = df_results.append({'classifiers': classifier.__class__.__name__,
                                        'fpr_train': fpr_train,
                                        'tpr_train': tpr_train,
                                        'auc_train': auc_train,
                                        'fpr_test': fpr_test,
                                        'tpr_test': tpr_test,
                                        'auc_test': auc_test}, ignore_index=True)
        df_results.set_index('classifiers', inplace=True)

        return classifier, pred_test, pred_train, df_results

    def get_baseline_model(self, x_train, y_train, classifiers, num_folds, random_state):
        """
        Test options and evaluation metric
        """
        self.logger.info("Running get baseline model function")
        num_folds = num_folds
        scoring = 'roc_auc'

        results = []
        names = []
        for name, cls in classifiers:
            kfold = StratifiedKFold(n_splits=num_folds, random_state=random_state)
            cv_results = cross_val_score(cls, x_train, y_train, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            self.logger.info(msg)
            # graph of both train and test ROC

        return names, results

    def leaf_analysis(self, df_train, df_test, target_col, pred_col, is_training, output_path):
        """
        Function to get the leaf node samples and probabilities
        df_train: training data set
        df_test: testing data set
        target_col: actual value of targetvar
        pred_col: predicted targetvar col
        """

        if is_training:
            grp_train = df_train.groupby([pred_col, target_col])['PatientID'].count().unstack(fill_value=0). \
                reset_index()
            grp_train.rename(columns={0: 'Train_0', 1: 'Train_1'}, inplace=True)
            # Total samples per leaf node
            grp_train['Total_Train'] = grp_train['Train_0'] + grp_train['Train_1']
            grp_train.rename(columns={pred_col: 'Train_Probability_1'}, inplace=True)
            grp_train['Train_Proportions'] = grp_train['Total_Train'] / sum(grp_train['Total_Train'])
            # pickle train leaf nodes for scoring purposes during the prediction process
            filepath = '{}/{}.pkl'.format(output_path, target_col + 'TrainLeafAnalysis')
            dump(grp_train, filepath)
        else:
            grp_train = df_train

        grp_test = df_test.groupby([pred_col, target_col])['PatientID'].count().unstack(fill_value=0). \
            reset_index()
        grp_test.rename(columns={0: 'Test_0', 1: 'Test_1'}, inplace=True)
        grp_test['Total_Test'] = grp_test['Test_0'] + grp_test['Test_1']
        grp_test['Test_Probability_1'] = grp_test['Test_1'] / grp_test['Total_Test']
        grp_test['Test_Proportions'] = grp_test['Total_Test'] / sum(grp_test['Total_Test'])
        df_leaf_results = grp_train.merge(grp_test, how='left', left_on='Train_Probability_1',
                                          right_on=pred_col)
        self.logger.info("--- Comparing the training probabilities to the test probabilities")
        df_leaf_results['Train_vs_Test'] = np.where(df_leaf_results['Test_Probability_1'] == 0, 0,
                                                    (df_leaf_results['Train_Probability_1'] /
                                                     df_leaf_results['Test_Probability_1']) * 100)
        self.logger.info("--- Saving leaf analysis results")
        df_leaf_results.drop(pred_col, axis=1, inplace=True)

        if is_training:
            filepath = '{}/{}.xlsx'.format(output_path, target_col + 'TestLeafAnalysis')
        else:
            filepath = '{}/{}.xlsx'.format(output_path, target_col + 'ValidationLeafAnalysis')
        df_leaf_results.to_excel(filepath)

        return df_leaf_results

    def post_prune_decision_tree(self, inner_tree, index, threshold):
        if inner_tree.value[index].min() < threshold:
            # turn node into a leaf by "unlinking" its children
            inner_tree.children_left[index] = TREE_LEAF
            inner_tree.children_right[index] = TREE_LEAF
        # if there are shildren, visit them as well
        if inner_tree.children_left[index] != TREE_LEAF:
            self.post_prune_decision_tree(inner_tree, inner_tree.children_left[index], threshold)
            self.post_prune_decision_tree(inner_tree, inner_tree.children_right[index], threshold)


class GraphingUtilClass:

    def __init__(self):
        self.logger = GlobalMemoryClass.LOGGER

    def draw_line_graph(self, list_x_series, list_y_series,
                        list_legend_names, list_legend_color,
                        x_label, y_label, title_text,
                        filename_html, tick_format, offline_plot=True):

        self.logger.info("Drawing line graph: {}".format(title_text))
        fig = go.Figure()

        for i in range(0, len(list_x_series)):
            fig.add_trace(go.Scatter(
                x=list_x_series[i],
                y=list_y_series[i],
                name=list_legend_names[i],
                mode='lines+markers',
                marker=dict(
                    size=10,
                    color=list_legend_color[i],
                ),
            ),
            )

        fig.update_layout(
            hoverlabel=dict(font=dict(family='sans-serif', size=20)),
            font=dict(
                family="Arial, Courier New, monospace",
                size=18,

            ),
            template="plotly_white",
            title=go.layout.Title(
                text=title_text,
                font=dict(
                    family="Arial, monospace",
                    size=30,
                    color="#10385c"
                )

            ),
            xaxis=go.layout.XAxis(
                title=go.layout.xaxis.Title(
                    text=x_label,
                    font=dict(
                        family="Arial, monospace",
                        size=18,
                        color="#10385c"
                    )
                )
            ))

        fig.update_yaxes(
            tickformat=tick_format,
            title=go.layout.yaxis.Title(
                text=y_label,
                font=dict(
                    family="Arial, monospace",
                    size=18,
                    color="#10385c"
                )
            )
        )

        if offline_plot:
            offline.plot(fig, filename=filename_html)

        return fig

    def draw_bar_graph(self, list_x_series, list_y_series,
                       list_legend_names, list_legend_color,
                       x_label, y_label, title_text,
                       filename_html, tick_format, offline_plot=True):

        self.logger.info("Drawing bar graph: {}".format(title_text))
        fig = go.Figure()

        for i in range(0, len(list_x_series)):
            fig.add_trace(go.Bar(
                x=list_x_series[i],
                y=list_y_series[i],
                name=list_legend_names[i],
                marker=dict(color=list_legend_color[i])
            ))

        fig.update_layout(
            template="plotly_white",
            title=go.layout.Title(
                text=title_text,
                font=dict(
                    family="Arial, monospace",
                    size=30,
                    color="#10385c"
                )

            ),
            xaxis=go.layout.XAxis(
                type='category',
                title=go.layout.xaxis.Title(
                    text=x_label,
                    font=dict(
                        family="Arial, monospace",
                        size=18,
                        color="#10385c"
                    )
                )
            ),
            yaxis=go.layout.YAxis(
                tickformat=tick_format,
                title=go.layout.yaxis.Title(
                    text=y_label,
                    font=dict(
                        family="Arial, monospace",
                        size=18,
                        color="#10385c"
                    )
                )
            )
        )
        if offline_plot:
            offline.plot(fig, filename=filename_html)

        return fig

    def correlation_heat_map(self, df):
        """
        Standard correlation heatmap
        """
        self.logger.info("Generating correlation heat map")
        fig, ax = plt.subplots(figsize=(18, 18))
        fig = sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
        plt.show()
        fig.figure.savefig('outputs/figures/correlations.png')

    def roc_plot(self, result_table, output_path, model_name, has_title=False):
        """
        Plot ROC curve, standard parameters. Need to amend to accept lists.
        """
        title_text = ''
        if has_title:
            title_text = 'ROC Curve Analysis for {} Classifier'.format(model_name)

        is_plotly = True
        if is_plotly:
            fig = go.Figure()
            for i in result_table.index:
                fig.add_trace(go.Scatter(
                    x=result_table.loc[i]['fpr_test'],
                    y=result_table.loc[i]['tpr_test'],
                    name='Test',
                    mode='lines',
                    marker=dict(
                        size=22,
                        color='#10385c'
                    ),
                ))

                fig.add_trace(go.Scatter(
                    x=result_table.loc[i]['fpr_train'],
                    y=result_table.loc[i]['tpr_train'],
                    name='Train',
                    mode='lines',
                    marker=dict(
                        size=22,
                        color='#f7941d'
                    ),
                ))

            fig.add_shape(
                # Line Diagonal
                type="line",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(
                    color="grey",
                    width=4,
                    dash="dot",
                ))

            fig.update_layout(
                hoverlabel=dict(font=dict(family='sans-serif', size=20)),
                font=dict(
                    family="Arial, Courier New, monospace",
                    size=18,

                ),
                template="plotly_white",
                title=go.layout.Title(
                    text=title_text,
                    font=dict(
                        family="Arial, monospace",
                        size=30,
                        color="#10385c"
                    )

                ),
                xaxis=go.layout.XAxis(
                    title=go.layout.xaxis.Title(
                        text="False Positive Rate",
                        font=dict(
                            family="Arial, monospace",
                            size=18,
                            color="#10385c"
                        )
                    )
                ),
                yaxis=go.layout.YAxis(

                    title=go.layout.yaxis.Title(
                        text="True Positive Rate",
                        font=dict(
                            family="Arial, monospace",
                            size=18,
                            color="#10385c"
                        )
                    )
                )
            )
            return fig
        else:
            self.logger.info("Running ROC plot function")
            for i in result_table.index:
                plt.plot(result_table.loc[i]['fpr_test'],
                         result_table.loc[i]['tpr_test'],
                         label="{}, AUC_Test={:.3f}".format(i, result_table.loc[i]['auc_test']))
                plt.plot(result_table.loc[i]['fpr_train'],
                         result_table.loc[i]['tpr_train'],
                         label="{}, AUC_Train={:.3f}".format(i, result_table.loc[i]['auc_train']))

            # plt.plot(fpr, tpr, label="AUC={:.3f}".format(auc))
            plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
            plt.xticks(np.arange(0, 1.1, step=0.1))
            plt.xlabel("False Positive Rate", fontsize=15)
            plt.yticks(np.arange(0, 1.1, step=0.1))
            plt.ylabel("True Positive Rate", fontsize=15)
            plt.title('ROC Curve Analysis for {} Classifier'.format(model_name), fontweight='bold', fontsize=15)
            plt.legend(prop={'size': 13}, loc='lower right')
            plt.show()
            plt.savefig('{}/{}_roc_curve.png'.format(output_path, model_name))
            plt.figure(figsize=(8, 6))
            return None

    def roc_train_test(self, target_col, y_train, pred_train, y_test, pred_test):
        """
        Plot ROC curve of train vs test.
        """
        self.logger.info("--- Metric: ROC curves")
        fpr_test, tpr_test, thresh_test = roc_curve(y_test, pred_test)
        auc_test = roc_auc_score(y_test, pred_test)
        test_gini = 2*auc_test-1
        fpr_train, tpr_train, thresh_train = roc_curve(y_train, pred_train)
        auc_train = roc_auc_score(y_train, pred_train)
        train_gini = 2*auc_train-1

        plt.subplots(figsize=(8, 6))
        plt.plot(fpr_test, tpr_test, label="{}, AUC_Test={:.3f}".format(target_col, auc_test, test_gini))
        plt.plot(fpr_train, tpr_train, label="{}, AUC_Train={:.3f}".format(target_col, auc_train, train_gini))
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xticks(np.arange(0, 1.1, step=0.1))
        plt.xlabel("False Positive Rate", fontsize=15)
        plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.ylabel("True Positive Rate", fontsize=15)
        plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
        plt.legend(loc=0)
        filepath = 'outputs/figures/{}.png'.format(target_col+"ROC")
        plt.savefig(filepath)
        plt.show()

    def roc_plot_array(self, target_col, auc_test, fpr_test, tpr_test):
        """
        Plot ROC curve, array parameters. Need to amend to accept lists.
        """
        self.logger.info("Running ROC plot function in array mode")
        plt.subplots(figsize=(8, 6))
        plt.plot(fpr_test,
                 tpr_test,
                 label="{}, AUC_Test={:.3f}".format(target_col, auc_test))

        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xticks(np.arange(0, 1.1, step=0.1))
        plt.xlabel("False Positive Rate", fontsize=15)
        plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.ylabel("True Positive Rate", fontsize=15)
        plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
        plt.legend(prop={'size': 13}, loc='lower right')
        # plt.figure(figsize=(8, 6))
        plt.savefig('outputs/figures/{}.png'.format(target_col))
        plt.show()

    def plot_cm(self, y_true, y_pred, normalize=False, title=None, draw_plotly=True):
        """
        self.logger.info and plot the confusion matrix. (directly from sklearn tutorial)
        Normalization can be applied by setting `normalize=True`.
        """
        self.logger.info("Running Confusion matrix plot function")
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        # classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            self.logger.info("Normalized confusion matrix")
        else:
            self.logger.info('Confusion matrix, without normalization')

        self.logger.info(cm)

        if draw_plotly:
            x = ['Positive', 'Negative']
            y = ['Positive', 'Negative']
            # change each element of z to type string for annotations
            cm_text = [[str(y) for y in x] for x in cm]

            # set up figure
            fig = ff.create_annotated_heatmap(cm, x=x, y=y, annotation_text=cm_text, colorscale='blues')
            fig['layout']['yaxis']['autorange'] = "reversed"
            # add title
            fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',
                              # xaxis = dict(title='x'),
                              # yaxis = dict(title='x')
                              )

            # add custom xaxis title
            fig.add_annotation(dict(font=dict(color="black", size=14),
                                    x=0.5,
                                    y=-0.15,
                                    showarrow=False,
                                    text="Predicted value",
                                    xref="paper",
                                    yref="paper"))

            # add custom yaxis title
            fig.add_annotation(dict(font=dict(color="black", size=14),
                                    x=-0.35,
                                    y=0.5,
                                    showarrow=False,
                                    text="Actual value",
                                    textangle=-90,
                                    xref="paper",
                                    yref="paper"))

            # adjust margins to make room for yaxis title
            fig.update_layout(margin=dict(t=50, l=200))

            offline.plot(fig, filename='../outputs/model_evaluation/confusion_matrix.html')

        else:
            fig, ax = plt.subplots()
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            ax.figure.colorbar(im, ax=ax)
            # We want to show all ticks...
            ax.set(title=title,
                   xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   # ... and label them with the respective list entries
                   # xticklabels=classes, yticklabels=classes,
                   ylabel='True label',
                   xlabel='Predicted label')

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")

            # Loop over data dimensions and create text annotations.
            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], fmt),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            fig.tight_layout()

            plt.show()

    def plot_feature_importance(self, model, df, has_title=False, draw_plotly=True):
        """
        Lets see what feature are important
        :return:
        """
        title_text = ''
        if has_title:
            title_text = 'Variable Importance'

        self.logger.info('Running Feature importance plot function')
        # Plot feature importance
        feature_importance = model.feature_importances_
        # make importance relative to max importance
        feature_importance_rel = 100.0 * (feature_importance / feature_importance.max())
        df_feature_importance = pd.DataFrame(feature_importance_rel)
        df_feature_importance.columns = ['Relative_Importance']
        df_feature_importance['Features'] = df.columns
        df_feature_importance['Importance'] = feature_importance
        df_feature_importance.sort_values(by='Importance', ascending=True, inplace=True)
        df_feature_importance = df_feature_importance[df_feature_importance['Relative_Importance'] > 0].copy()
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5

        if draw_plotly:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(y=df_feature_importance['Features'],
                       x=df_feature_importance['Relative_Importance'],
                       orientation='h',
                       name='Observed',
                       marker=dict(color='#10385c')
                       ))

            fig.update_layout(
                template="plotly_white",
                title=go.layout.Title(
                    text=title_text,
                    font=dict(
                        family="Arial, monospace",
                        size=30,
                        color="#10385c"
                    )

                ),
                xaxis=go.layout.XAxis(
                    title=go.layout.xaxis.Title(
                        text="Relative Importance",
                        font=dict(
                            family="Arial, monospace",
                            size=18,
                            color="#10385c"
                        )
                    )
                ),
                yaxis=go.layout.YAxis(
                    title=go.layout.yaxis.Title(
                        font=dict(
                            family="Arial, monospace",
                            size=18,
                            color="#10385c"
                        )
                    )
                )
            )
            return fig
        else:
            plt.subplot(1, 2, 2)
            plt.barh(pos, feature_importance[sorted_idx], align='center')
            plt.yticks(pos, df.columns[sorted_idx])  # boston.feature_names[sorted_idx])
            plt.xlabel('Relative Importance')
            plt.title('Variable Importance')
            plt.show()
            # plt.savefig('{}/{}_feature_importance.png'.format(output_path, model_name))
            return plt

    # def visualize_tree(self, cls, list_features, list_targets, output_path, matplot=True):
    #     # Create DOT data
    #     self.logger.info("Visualizing Decision Tree")
    #
    #     if matplot:
    #         tree.plot_tree(cls)
    #
    #         fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)
    #         tree.plot_tree(cls, filled=True)
    #         fig.savefig("{}/decision_tree.png".format(output_path))
    #     else:
    #         self.logger.info("Create DOT data")
    #         dot_data = tree.export_graphviz(cls, feature_names=list_features, class_names=list_targets, out_file=None,
    #                                         rounded=True, max_depth=7)
    #
    #         self.logger.info("raw graph")
    #         # Draw graph
    #         graph = pydotplus.graph_from_dot_data(dot_data)
    #
    #         self.logger.info("Show graph")
    #         # Show graph
    #         Image(graph.create_png())
    #
    #         self.logger.info("Create PNG")
    #         # Create PNG
    #         graph.write_png("{}/decision_tree.png".format(output_path))

    def decile_analysis_plot(self, df, target_col, pred_col, id_col, num_bins, class_weight, has_title=False, draw_plotly=True):
        """
        Function to run the decile view to assess model performance
        target_col: name of target variable column
        pred_col: name of predicted variable column
        id_col: name of column used to identify each record of dataframe
        file_path: path where figure needs to be saved
        """
        title_text = ''
        if has_title:
            title_text = target_col + " performance"

        df = df.sort_values(by=[pred_col], ascending=False)
        df = df.reset_index(drop=True)
        bins = list(range(0, num_bins + 1))  # [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        bin_size = df.shape[0] / num_bins
        bins = np.array(bins, dtype=float) * bin_size
        labels = list(range(1, num_bins + 1))
        df['binned'] = pd.cut(df.index, bins=bins, labels=labels)
        df_grp = df.groupby(['binned']).agg({pred_col: 'mean', id_col: 'count', target_col: 'sum'})
        df_grp['Predicted_Count'] = df_grp[pred_col] * df_grp[id_col]
        df_grp.rename(columns={'PatientID': 'Counts', target_col: 'Observed_Count'}, inplace=True)
        df_grp['Predicted'] = (df_grp['Predicted_Count'] / bin_size) * 100 * class_weight
        df_grp['Observed'] = (df_grp['Observed_Count'] / bin_size) * 100

        self.logger.info("---DRAWING DECILE PLOT")
        if draw_plotly:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(x=labels,
                       y=df_grp['Observed'],
                       name='Observed',
                       marker=dict(color='#10385c')
                       ))
            fig.add_trace(
                go.Bar(x=labels,
                       y=df_grp['Predicted'],
                       name='Predicted',
                       marker=dict(color='#f7941d')
                       ))

            fig.update_layout(
                template="plotly_white",
                title=go.layout.Title(
                    text=title_text,
                    font=dict(
                        family="Arial, monospace",
                        size=30,
                        color="#10385c"
                    )

                ),
                xaxis=go.layout.XAxis(
                    title=go.layout.xaxis.Title(
                        text="Deciles",
                        font=dict(
                            family="Arial, monospace",
                            size=18,
                            color="#10385c"
                        )
                    )
                ),
                yaxis=go.layout.YAxis(
                    title=go.layout.yaxis.Title(
                        text="Percentage",
                        font=dict(
                            family="Arial, monospace",
                            size=18,
                            color="#10385c"
                        )
                    )
                )
            )

            return fig, df_grp
        else:
            ax = df_grp[['Predicted', 'Observed']].plot(kind='bar',
                                                        title=title_text,
                                                        figsize=(15, 8),
                                                        legend=True,
                                                        fontsize=12,
                                                        color=['salmon', 'darkblue'],
                                                        grid=True)

            ax.set_xlabel('Decile', fontsize=12)
            ax.set_ylabel('Percentage', fontsize=12)
            # plt.savefig('{}/{}.png'.format(output_path, target_col + "Deciles"))
            plt.show()
            return plt, df_grp

    def plot_lift_curve(self, df, target_col, pred_col, id_col, class_weight, draw_plotly=True, has_title=True):
        """
        Function to run the lift curve
        target_col: name of target variable column
        pred_col: name of predicted variable column
        id_col: name of column used to identify each record of dataframe
        file_path: path where figure needs to be saved
        """
        title_text = ''
        if has_title:
            title_text = 'Lift Curve for {}'.format(target_col.replace("_Actual", ""))

        df = df.sort_values(by=[pred_col], ascending=False)
        df = df.reset_index(drop=True)
        num_bins = 100
        bins = list(range(0, num_bins + 1))  # [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        bin_size = df.shape[0] / num_bins
        bins = np.array(bins, dtype=float) * bin_size
        labels = list(range(1, num_bins + 1))  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        df['binned'] = pd.cut(df.index, bins=bins, labels=labels)
        df_grp = df.groupby(['binned']).agg({pred_col: 'mean', id_col: 'count', target_col: 'sum'})
        df_grp['Expected_Count'] = df_grp[pred_col] * df_grp[id_col]
        df_grp.rename(columns={'PatientID': 'Counts', target_col: 'Observed_Count'}, inplace=True)
        df_grp['Expected'] = (df_grp['Expected_Count'] / bin_size) * 100 * class_weight
        df_grp['Observed'] = (df_grp['Observed_Count'] / bin_size) * 100
        df_grp['CumulativeExpected'] = df_grp['Expected_Count'].cumsum()

        sum_expected = df_grp['Expected_Count'].sum()
        df_grp['Gain'] = df_grp['CumulativeExpected'] / sum_expected
        df_grp['Bins_Percentage'] = [b / max(labels) for b in labels]
        df_grp['Cumulative_Lift'] = df_grp['Gain'] / df_grp['Bins_Percentage']
        df_grp['Cumulative_Random_Lift'] = 1

        self.logger.info("PlOTTING LIFT CURVE")
        if draw_plotly:
            fig_lift = go.Figure()
            fig_lift.add_trace(go.Scatter(
                x=labels,
                y=df_grp['Cumulative_Lift'],
                name='Lift (Model)',
                mode='lines+markers',
                marker=dict(
                    size=5,
                    color='#10385c'
                ),
            ))

            fig_lift.add_trace(go.Scatter(
                x=labels,
                y=df_grp['Cumulative_Random_Lift'],
                name='Lift (Random)',
                mode='lines+markers',
                marker=dict(
                    size=5,
                    color='#eb0e48'
                ),
            ))

            fig_lift.update_layout(
                hoverlabel=dict(font=dict(family='sans-serif', size=20)),
                font=dict(
                    family="Arial, Courier New, monospace",
                    size=18,

                ),
                template="plotly_white",
                title=go.layout.Title(
                    text=title_text,
                    font=dict(
                        family="Arial, monospace",
                        size=30,
                        color="#10385c"
                    )

                ),
                xaxis=go.layout.XAxis(
                    title=go.layout.xaxis.Title(
                        text='Proportion of sample',
                        font=dict(
                            family="Arial, monospace",
                            size=18,
                            color="#10385c"
                        )
                    )
                ),
                yaxis=go.layout.YAxis(

                    title=go.layout.yaxis.Title(
                        text='Lift',
                        font=dict(
                            family="Arial, monospace",
                            size=18,
                            color="#10385c"
                        )
                    )
                )
            )

            self.logger.info("PlOTTING GAIN CURVE")
            fig_gain = go.Figure()
            fig_gain.add_trace(go.Scatter(
                x=labels,
                y=df_grp['Gain'],
                name='% cumulative events (model)',
                mode='lines+markers',
                marker=dict(
                    size=5,
                    color='#10385c'
                ),
            ))

            fig_gain.add_trace(go.Scatter(
                x=labels,
                y=df_grp['Bins_Percentage'],
                name='% cumulative events (random)',
                mode='lines+markers',
                marker=dict(
                    size=5,
                    color='#eb0e48'
                ),
            ))

            fig_gain.update_layout(
                hoverlabel=dict(font=dict(family='sans-serif', size=20)),
                font=dict(
                    family="Arial, Courier New, monospace",
                    size=18,

                ),
                template="plotly_white",
                title=go.layout.Title(
                    text='Gain Chart for {}'.format(target_col.replace("_Actual", "")),
                    font=dict(
                        family="Arial, monospace",
                        size=30,
                        color="#10385c"
                    )

                ),
                xaxis=go.layout.XAxis(

                    title=go.layout.xaxis.Title(
                        text='Proportion of sample',
                        font=dict(
                            family="Arial, monospace",
                            size=18,
                            color="#10385c"
                        )
                    )
                ),
                yaxis=go.layout.YAxis(
                    tickformat=',.0%',
                    title=go.layout.yaxis.Title(
                        text='Gain Chart',
                        font=dict(
                            family="Arial, monospace",
                            size=18,
                            color="#10385c"
                        )
                    )
                )
            )
            return fig_lift, fig_gain
        else:
            return None, None
        # # Plot the figure
        #  fig, axis = plt.subplots()
        #  fig.figsize = (40,40)
        #  axis.plot(x_val, y_v, 'g-', linewidth = 3, markersize = 5)
        #  axis.plot(x_val, np.ones(len(x_val)), 'k-')
        #  axis.set_xlabel('Proportion of sample')
        #  axis.set_ylabel('Lift')
        #  plt.title('Lift Curve')
        #  plt.show()


class ParameterTuningClass:

    def __init__(self, x_train, y_train, model, hyperparameters):
        self.x_train = x_train
        self.y_train = y_train
        self.model = model
        self.hyperparameters = hyperparameters
        self.logger = GlobalMemoryClass.LOGGER

    def grid_search(self, cv_count=4):
        # Create randomized search 4-fold cross validation and 100 iterations
        cv = cv_count
        clf = GridSearchCV(self.model,
                           self.hyperparameters,
                           cv=cv,
                           verbose=0,
                           n_jobs=-1,
                           )
        # Fit randomized search
        best_model = clf.fit(self.x_train, self.y_train)
        message = (best_model.best_score_, best_model.best_params_)
        self.logger.info("Best: %f using %s" % message)

        return best_model, best_model.best_params_

    def best_model_predict(self, x_test):
        best_model, _ = self.grid_search()
        pred = best_model.predict(x_test)
        return pred
